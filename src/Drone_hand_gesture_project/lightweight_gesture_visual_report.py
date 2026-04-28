import os
import pickle
from dataclasses import dataclass

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


@dataclass
class GestureEnhancementResult:
    enhanced_features: np.ndarray
    joint_attention: np.ndarray
    channel_attention: np.ndarray
    fused_features: np.ndarray


class LightweightGestureAttention:
    """轻量化关键点注意力模块。"""

    def __init__(self):
        self.channel_kernel = np.array([1.15, 1.00, 0.85], dtype=np.float32)

    def enhance(self, sample: np.ndarray) -> GestureEnhancementResult:
        joints = sample.reshape(21, 3).astype(np.float32)
        wrist = joints[0:1]
        relative = joints - wrist

        joint_energy = np.linalg.norm(relative, axis=1)
        joint_attention = self._softmax(0.9 * joint_energy + self._finger_prior())
        channel_attention = self._sigmoid(relative.std(axis=0) * self.channel_kernel + np.array([0.2, 0.1, -0.1], dtype=np.float32))

        attended = relative * joint_attention[:, None] * channel_attention[None, :]
        fused = 0.65 * relative + 0.55 * attended
        enhanced = fused + wrist
        return GestureEnhancementResult(
            enhanced_features=enhanced.reshape(-1),
            joint_attention=joint_attention,
            channel_attention=channel_attention,
            fused_features=fused.reshape(21, 3),
        )

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _finger_prior() -> np.ndarray:
        return np.array([
            0.10, 0.18, 0.22, 0.26, 0.30,
            0.16, 0.24, 0.28, 0.32,
            0.14, 0.20, 0.26, 0.32,
            0.12, 0.18, 0.24, 0.30,
            0.12, 0.16, 0.22, 0.28,
        ], dtype=np.float32)


class PrototypeGestureClassifier:
    def __init__(self, prototypes: np.ndarray, class_names):
        self.prototypes = prototypes.astype(np.float32)
        self.class_names = list(class_names)

    @classmethod
    def from_dataset(cls, features: np.ndarray, labels: np.ndarray, class_names):
        prototypes = []
        for class_idx in range(len(class_names)):
            prototypes.append(features[labels == class_idx].mean(axis=0))
        return cls(np.vstack(prototypes), class_names)

    def predict_confidence(self, sample: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(self.prototypes - sample[None, :], axis=1)
        logits = -distances / (np.std(distances) + 1e-6)
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        return probs / probs.sum()


def load_dataset(dataset_path: str):
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
    return (
        np.asarray(dataset["features"], dtype=np.float32),
        np.asarray(dataset["labels"], dtype=np.int64),
        dataset["gesture_classes"],
    )


def plot_hand(ax, sample: np.ndarray, title: str, attention: np.ndarray | None = None):
    joints = sample.reshape(21, 3)
    x = joints[:, 0]
    y = -joints[:, 1]
    colors = attention if attention is not None else np.linspace(0.2, 1.0, len(x))
    for start, end in HAND_CONNECTIONS:
        ax.plot([x[start], x[end]], [y[start], y[end]], color="#7aa6ff", linewidth=1.4, alpha=0.8)
    scatter = ax.scatter(x, y, c=colors, cmap="turbo", s=52, edgecolors="black", linewidths=0.3)
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    return scatter


def build_report(output_dir: str):
    dataset_path = os.path.join(os.path.dirname(__file__), "gesture_dataset.pkl")
    features, labels, class_names = load_dataset(dataset_path)
    enhancer = LightweightGestureAttention()

    enhanced_features = []
    attention_bank = []
    channel_bank = []
    fused_bank = []
    for sample in features:
        result = enhancer.enhance(sample)
        enhanced_features.append(result.enhanced_features)
        attention_bank.append(result.joint_attention)
        channel_bank.append(result.channel_attention)
        fused_bank.append(result.fused_features)

    enhanced_features = np.asarray(enhanced_features, dtype=np.float32)
    attention_bank = np.asarray(attention_bank, dtype=np.float32)
    channel_bank = np.asarray(channel_bank, dtype=np.float32)
    fused_bank = np.asarray(fused_bank, dtype=np.float32)

    raw_classifier = PrototypeGestureClassifier.from_dataset(features, labels, class_names)
    enhanced_classifier = PrototypeGestureClassifier.from_dataset(enhanced_features, labels, class_names)

    os.makedirs(output_dir, exist_ok=True)
    selected_indices = [np.where(labels == class_idx)[0][0] for class_idx in range(len(class_names))]

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.1])

    ax0 = fig.add_subplot(gs[0, 0])
    raw_attention = attention_bank[selected_indices[0]]
    plot_hand(ax0, features[selected_indices[0]], f"Raw Gesture: {class_names[0]}", raw_attention)

    ax1 = fig.add_subplot(gs[0, 1])
    plot_hand(ax1, enhanced_features[selected_indices[0]], "Enhanced Landmark Layout", raw_attention)

    ax2 = fig.add_subplot(gs[0, 2])
    mean_joint_attention = attention_bank.mean(axis=0)
    ax2.bar(np.arange(21), mean_joint_attention, color=plt.cm.viridis(mean_joint_attention / mean_joint_attention.max()))
    ax2.set_title("Joint Attention Weights", fontsize=12, weight="bold")
    ax2.set_xlabel("Landmark Index")
    ax2.set_ylabel("Weight")

    ax3 = fig.add_subplot(gs[1, 0])
    mean_channel_attention = channel_bank.mean(axis=0)
    ax3.bar(["x", "y", "z"], mean_channel_attention, color=["#ff8c69", "#4ecdc4", "#6c5ce7"])
    ax3.set_ylim(0, 1.1)
    ax3.set_title("Channel Attention", fontsize=12, weight="bold")

    ax4 = fig.add_subplot(gs[1, 1])
    fused_strength = np.linalg.norm(fused_bank, axis=2)
    heat = ax4.imshow(fused_strength[:60], aspect="auto", cmap="magma")
    ax4.set_title("Feature Reweighting Heatmap", fontsize=12, weight="bold")
    ax4.set_xlabel("Landmark Index")
    ax4.set_ylabel("Sample Index")
    fig.colorbar(heat, ax=ax4, fraction=0.046, pad=0.04)

    ax5 = fig.add_subplot(gs[1, 2])
    confidence_rows = []
    before_scores = []
    after_scores = []
    for idx in selected_indices:
        raw_probs = raw_classifier.predict_confidence(features[idx])
        enhanced_probs = enhanced_classifier.predict_confidence(enhanced_features[idx])
        confidence_rows.append(np.vstack([raw_probs, enhanced_probs]))
        before_scores.append(raw_probs.max())
        after_scores.append(enhanced_probs.max())
    conf_heat = np.concatenate(confidence_rows, axis=0)
    im = ax5.imshow(conf_heat, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=max(0.6, conf_heat.max()))
    ax5.set_title("Confidence Recalibration", fontsize=12, weight="bold")
    ax5.set_yticks(np.arange(len(selected_indices) * 2))
    ax5.set_yticklabels([f"{name} raw" if i % 2 == 0 else f"{name} enh" for name in class_names for i in range(2)], fontsize=8)
    ax5.set_xticks(np.arange(len(class_names)))
    ax5.set_xticklabels(class_names, rotation=30, ha="right")
    fig.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)

    ax6 = fig.add_subplot(gs[2, :2])
    pca_input = enhanced_features - enhanced_features.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(pca_input, full_matrices=False)
    coords = pca_input @ vt[:2].T
    colors = plt.cm.tab10(labels / max(1, labels.max()))
    ax6.scatter(coords[:, 0], coords[:, 1], c=colors, s=22, alpha=0.72)
    for class_idx, name in enumerate(class_names):
        class_coords = coords[labels == class_idx]
        center = class_coords.mean(axis=0)
        ax6.text(center[0], center[1], name, fontsize=10, weight="bold")
    ax6.set_title("Enhanced Gesture Embedding Distribution", fontsize=12, weight="bold")
    ax6.set_xlabel("Principal Component 1")
    ax6.set_ylabel("Principal Component 2")

    ax7 = fig.add_subplot(gs[2, 2])
    width = 0.35
    x = np.arange(len(class_names))
    ax7.bar(x - width / 2, before_scores, width, label="Before", color="#74b9ff")
    ax7.bar(x + width / 2, after_scores, width, label="After", color="#55efc4")
    ax7.set_ylim(0, 1.05)
    ax7.set_xticks(x)
    ax7.set_xticklabels(class_names, rotation=30, ha="right")
    ax7.set_title("Top Confidence Gain", fontsize=12, weight="bold")
    ax7.legend()

    fig.suptitle("Lightweight Neural Gesture Enhancement and Visualization Report", fontsize=18, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    report_path = os.path.join(output_dir, "gesture_enhancement_report.png")
    fig.savefig(report_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    module_fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_hand(axes[0, 0], features[selected_indices[2]], f"Sample: {class_names[2]}", attention_bank[selected_indices[2]])
    plot_hand(axes[0, 1], enhanced_features[selected_indices[2]], "Attention-Reweighted Gesture", attention_bank[selected_indices[2]])
    axes[1, 0].plot(np.arange(21), attention_bank[selected_indices[2]], marker="o", color="#e17055")
    axes[1, 0].set_title("Spatial Joint Attention Curve", weight="bold")
    axes[1, 0].set_xlabel("Landmark Index")
    axes[1, 0].set_ylabel("Attention Weight")
    axes[1, 1].bar(["x", "y", "z"], channel_bank[selected_indices[2]], color=["#fdcb6e", "#00b894", "#6c5ce7"])
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_title("Axis Channel Attention", weight="bold")
    module_fig.suptitle("Gesture Module-Level Visualization", fontsize=17, weight="bold")
    module_fig.tight_layout(rect=[0, 0, 1, 0.96])
    module_path = os.path.join(output_dir, "gesture_module_views.png")
    module_fig.savefig(module_path, dpi=220, bbox_inches="tight")
    plt.close(module_fig)

    return report_path, module_path


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "visual_reports")
    report_path, module_path = build_report(output_dir)
    print(f"✅ 手势增强总报告已保存: {report_path}")
    print(f"✅ 手势模块级可视化已保存: {module_path}")


if __name__ == "__main__":
    main()
