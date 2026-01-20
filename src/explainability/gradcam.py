from __future__ import annotations
import numpy as np
import tensorflow as tf

def _normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32)
    x = np.maximum(x, 0.0)
    return x / (x.max() + eps)

def gradcam_mel(model: tf.keras.Model, x: dict[str, np.ndarray], conv_layer_name: str = "mel_conv_last") -> np.ndarray:
    mel = tf.convert_to_tensor(x["mel_in"])
    seq = tf.convert_to_tensor(x["seq_in"])

    conv_layer = model.get_layer(conv_layer_name)
    grad_model = tf.keras.Model(inputs=model.inputs, outputs=[conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_out, y = grad_model({"mel_in": mel, "seq_in": seq}, training=False)
        p_fake = y[0, 0]
        tape.watch(conv_out)

    grads = tape.gradient(p_fake, conv_out)           # [1,H,W,C]
    weights = tf.reduce_mean(grads, axis=(1, 2))      # [1,C]
    cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)  # [1,H,W]
    cam = cam[0].numpy()
    cam = _normalize(cam)

    # Upsample to mel size [80,T]
    target_h = x["mel_in"].shape[1]
    target_w = x["mel_in"].shape[2]
    cam_up = tf.image.resize(cam[..., None], (target_h, target_w), method="bilinear").numpy()[:, :, 0]
    return _normalize(cam_up)
