import os
import argparse
import base64
import json
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.io import write_video

import librosa
import numpy as np
import requests
import gradio as gr
from PIL import Image
from tqdm import tqdm

from emage_utils.motion_io import beat_format_save
from emage_utils import fast_render
from models.emage_audio import EmageAudioModel, EmageVQVAEConv, EmageVAEConv, EmageVQModel

OUTPUT_ROOT = Path("./outputs")


@dataclass
class QwenOmniResponse:
    text: str
    audio_path: str | None


def inference(model, motion_vq, audio_path, device, save_folder, sr, pose_fps):
    audio, _ = librosa.load(audio_path, sr=sr)
    audio = torch.from_numpy(audio).to(device).unsqueeze(0)
    speaker_id = torch.zeros(1, 1).long().to(device)
    print(f"audio shape: {tuple(audio.shape)}, speaker_id shape: {tuple(speaker_id.shape)}")
    with torch.no_grad():
        trans = torch.zeros(1, 1, 3).to(device)

        latent_dict = model.inference(audio, speaker_id, motion_vq, masked_motion=None, mask=None)
        print({k: (v.shape if isinstance(v, torch.Tensor) else None) for k, v in latent_dict.items()})

        face_latent = latent_dict["rec_face"] if model.cfg.lf > 0 and model.cfg.cf == 0 else None
        upper_latent = latent_dict["rec_upper"] if model.cfg.lu > 0 and model.cfg.cu == 0 else None
        hands_latent = latent_dict["rec_hands"] if model.cfg.lh > 0 and model.cfg.ch == 0 else None
        lower_latent = latent_dict["rec_lower"] if model.cfg.ll > 0 and model.cfg.cl == 0 else None

        face_index = torch.max(F.log_softmax(latent_dict["cls_face"], dim=2), dim=2)[1] if model.cfg.cf > 0 else None
        upper_index = torch.max(F.log_softmax(latent_dict["cls_upper"], dim=2), dim=2)[1] if model.cfg.cu > 0 else None
        hands_index = torch.max(F.log_softmax(latent_dict["cls_hands"], dim=2), dim=2)[1] if model.cfg.ch > 0 else None
        lower_index = torch.max(F.log_softmax(latent_dict["cls_lower"], dim=2), dim=2)[1] if model.cfg.cl > 0 else None

        print(
            "latents shapes:",
            "face",
            None if face_latent is None else tuple(face_latent.shape),
            "upper",
            None if upper_latent is None else tuple(upper_latent.shape),
            "hands",
            None if hands_latent is None else tuple(hands_latent.shape),
            "lower",
            None if lower_latent is None else tuple(lower_latent.shape),
        )
        print(
            "indices shapes:",
            "face",
            None if face_index is None else tuple(face_index.shape),
            "upper",
            None if upper_index is None else tuple(upper_index.shape),
            "hands",
            None if hands_index is None else tuple(hands_index.shape),
            "lower",
            None if lower_index is None else tuple(lower_index.shape),
        )

        all_pred = motion_vq.decode(
            face_latent=face_latent,
            upper_latent=upper_latent,
            lower_latent=lower_latent,
            hands_latent=hands_latent,
            face_index=face_index,
            upper_index=upper_index,
            lower_index=lower_index,
            hands_index=hands_index,
            get_global_motion=True,
            ref_trans=trans[:, 0],
        )
        print({k: (v.shape if isinstance(v, torch.Tensor) else None) for k, v in all_pred.items()})

    motion_pred = all_pred["motion_axis_angle"]
    t = motion_pred.shape[1]
    motion_pred = motion_pred.cpu().numpy().reshape(t, -1)
    face_pred = all_pred["expression"].cpu().numpy().reshape(t, -1)
    trans_pred = all_pred["trans"].cpu().numpy().reshape(t, -1)

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    np.save(os.path.join(save_folder, f"{base_name}_face_pred.npy"), face_pred)
    if face_latent is not None:
        np.save(os.path.join(save_folder, f"{base_name}_face_latent.npy"), face_latent.detach().cpu().numpy())

    print(f"motion_pred: {motion_pred.shape}, face_pred: {face_pred.shape}, trans_pred: {trans_pred.shape}")
    beat_format_save(
        os.path.join(save_folder, f"{base_name}_output.npz"),
        motion_pred,
        upsample=30 // pose_fps,
        expressions=face_pred,
        trans=trans_pred,
    )
    return t


def visualize_one(save_folder, audio_path, nopytorch3d=False):
    npz_path = os.path.join(save_folder, f"{os.path.splitext(os.path.basename(audio_path))[0]}_output.npz")
    motion_dict = np.load(npz_path, allow_pickle=True)
    print({k: (v.shape if isinstance(v, np.ndarray) else type(v)) for k, v in motion_dict.items()})
    if not nopytorch3d:
        from emage_utils.npz2pose import render2d

        v2d_face = render2d(motion_dict, (512, 512), face_only=True, remove_global=True)
        write_video(npz_path.replace(".npz", "_2dface.mp4"), v2d_face.permute(0, 2, 3, 1), fps=30)
        fast_render.add_audio_to_video(
            npz_path.replace(".npz", "_2dface.mp4"),
            audio_path,
            npz_path.replace(".npz", "_2dface_audio.mp4"),
        )
        v2d_body = render2d(motion_dict, (720, 480), face_only=False, remove_global=True)
        write_video(npz_path.replace(".npz", "_2dbody.mp4"), v2d_body.permute(0, 2, 3, 1), fps=30)
        fast_render.add_audio_to_video(
            npz_path.replace(".npz", "_2dbody.mp4"),
            audio_path,
            npz_path.replace(".npz", "_2dbody_audio.mp4"),
        )
    fast_render.render_one_sequence_with_face(
        npz_path,
        os.path.dirname(npz_path),
        audio_path,
        model_folder="./emage_evaltools/smplx_models/",
    )


def save_numpy_audio(audio_tuple, target_path):
    if audio_tuple is None:
        return None
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    sample_rate, data = audio_tuple
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = np.clip(data, -1.0, 1.0)
    int_data = (data * 32767).astype(np.int16)
    with wave.open(target_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(int_data.tobytes())
    return target_path


def encode_image_numpy(image_np):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(image_np.astype("uint8"))
    with Path(OUTPUT_ROOT, "tmp_image.jpg").open("wb") as image_file:
        image.save(image_file, format="JPEG")
        image_file.seek(0)
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def encode_audio_base64(audio_path):
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")


def qwen3_omni_chat(user_text, camera_frame, audio_path):
    api_key = os.getenv("QWEN_OMNI_API_KEY")
    if not api_key:
        raise ValueError("Missing QWEN_OMNI_API_KEY environment variable.")
    base_url = os.getenv("QWEN_OMNI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = os.getenv("QWEN_OMNI_MODEL", "qwen3-omni")

    content = []
    if user_text:
        content.append({"type": "text", "text": user_text})
    if camera_frame is not None:
        content.append({"type": "image_url", "image_url": {"url": encode_image_numpy(camera_frame)}})
    if audio_path is not None:
        content.append(
            {
                "type": "input_audio",
                "input_audio": {"data": encode_audio_base64(audio_path), "format": "wav"},
            }
        )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是一个多模态数字人助手，请根据音频和视频信息与用户自然交流。",
            },
            {"role": "user", "content": content},
        ],
    }

    response = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()

    message = data["choices"][0]["message"]
    content = message.get("content", "")
    if isinstance(content, list):
        text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    else:
        text = content if isinstance(content, str) else ""
    audio_path_out = None
    audio_blob = message.get("audio")
    if isinstance(audio_blob, dict) and "data" in audio_blob:
        audio_output = base64.b64decode(audio_blob["data"])
        audio_path_out = str(Path(OUTPUT_ROOT, f"qwen_reply_{int(time.time())}.wav"))
        with open(audio_path_out, "wb") as audio_file:
            audio_file.write(audio_output)
    return QwenOmniResponse(text=text, audio_path=audio_path_out)


def load_emage_models(device):
    face_motion_vq = EmageVQVAEConv.from_pretrained("/workspace/data/models/emage_audio", subfolder="emage_vq/face").to(device)
    upper_motion_vq = EmageVQVAEConv.from_pretrained("/workspace/data/models/emage_audio", subfolder="emage_vq/upper").to(device)
    lower_motion_vq = EmageVQVAEConv.from_pretrained("/workspace/data/models/emage_audio", subfolder="emage_vq/lower").to(device)
    hands_motion_vq = EmageVQVAEConv.from_pretrained("/workspace/data/models/emage_audio", subfolder="emage_vq/hands").to(device)
    global_motion_ae = EmageVAEConv.from_pretrained("/workspace/data/models/emage_audio", subfolder="emage_vq/global").to(device)
    motion_vq = EmageVQModel(
        face_model=face_motion_vq,
        upper_model=upper_motion_vq,
        lower_model=lower_motion_vq,
        hands_model=hands_motion_vq,
        global_model=global_motion_ae,
    ).to(device)
    motion_vq.eval()

    model = EmageAudioModel.from_pretrained("/workspace/data/models/emage_audio").to(device)
    model.eval()
    return model, motion_vq


def find_latest_video(save_folder, stem):
    candidates = list(Path(save_folder).glob(f"{stem}*.mp4"))
    if not candidates:
        candidates = list(Path(save_folder).glob("*.mp4"))
    if not candidates:
        raise FileNotFoundError("No rendered video was found in the output folder.")
    return str(max(candidates, key=lambda path: path.stat().st_mtime))


def render_face_video(save_folder, audio_path):
    npz_path = os.path.join(save_folder, f"{Path(audio_path).stem}_output.npz")
    fast_render.render_one_sequence_with_face(
        npz_path,
        save_folder,
        audio_path,
        model_folder="./emage_evaltools/smplx_models/",
    )
    return find_latest_video(save_folder, Path(audio_path).stem)


def generate_digital_human_video(model, motion_vq, audio_path, device):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    sr, pose_fps = model.cfg.audio_sr, model.cfg.pose_fps
    inference(model, motion_vq, audio_path, device, str(OUTPUT_ROOT), sr, pose_fps)
    return render_face_video(str(OUTPUT_ROOT), audio_path)


def build_gradio_app(model, motion_vq, device):
    with gr.Blocks() as demo:
        gr.Markdown("# OmniTalker - Qwen3-Omni 端到端数字人")
        with gr.Row():
            camera = gr.Image(sources=["webcam"], type="numpy", label="摄像头")
            microphone = gr.Audio(sources=["microphone"], type="numpy", label="麦克风")
        with gr.Row():
            user_text = gr.Textbox(lines=2, label="补充文本（可选）")
        with gr.Row():
            response_text = gr.Textbox(lines=3, label="Qwen3-Omni 回复")
        with gr.Row():
            avatar_video = gr.Video(label="数字人")

        run_button = gr.Button("开始对话")
        status = gr.Markdown()
        error = gr.Markdown()

        def run_chat(camera_frame, mic_audio, typed_text):
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            mic_path = None
            if mic_audio is not None:
                mic_path = str(Path(OUTPUT_ROOT, f"user_{int(time.time())}.wav"))
                save_numpy_audio(mic_audio, mic_path)

            try:
                response = qwen3_omni_chat(typed_text, camera_frame, mic_path)
            except requests.HTTPError as exc:
                return None, "", "API 调用失败", f"API error: {exc}"
            except ValueError as exc:
                return None, "", "缺少 API Key", str(exc)

            reply_audio = response.audio_path or mic_path
            if reply_audio is None:
                return None, response.text, "缺少音频输入", "未生成音频，无法驱动数字人。"

            video_path = generate_digital_human_video(model, motion_vq, reply_audio, device)
            return video_path, response.text, "完成", ""

        run_button.click(
            run_chat,
            inputs=[camera, microphone, user_text],
            outputs=[avatar_video, response_text, status, error],
        )
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_folder", type=str, default="./examples/audio")
    parser.add_argument("--save_folder", type=str, default="./examples/motion")
    parser.add_argument("--visualization", action="store_true")
    parser.add_argument("--nopytorch3d", action="store_true")
    parser.add_argument("--gradio", action="store_true")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, motion_vq = load_emage_models(device)

    if args.gradio:
        demo = build_gradio_app(model, motion_vq, device)
        demo.launch(server_name=args.host, server_port=args.port, share=args.share)
        return

    os.makedirs(args.save_folder, exist_ok=True)
    audio_files = [os.path.join(args.audio_folder, f) for f in os.listdir(args.audio_folder) if f.endswith(".wav")]
    sr, pose_fps = model.cfg.audio_sr, model.cfg.pose_fps
    print(f"sr={sr}, pose_fps={pose_fps}")
    all_t = 0
    start_time = time.time()

    for audio_path in tqdm(audio_files, desc="Inference"):
        all_t += inference(model, motion_vq, audio_path, device, args.save_folder, sr, pose_fps)
        if args.visualization:
            visualize_one(args.save_folder, audio_path, args.nopytorch3d)
    print(f"generate total {all_t / pose_fps:.2f} seconds motion in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
