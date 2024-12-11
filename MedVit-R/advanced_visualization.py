import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix


class ViTVisualizer:
    def __init__(self, model, device, class_names):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.model.eval()

    def get_attention_weights(self, x):
        """Extract attention weights from the last attention layer"""
        self.model.eval()
        with torch.no_grad():
            # Forward pass through PatchEmbed (handled inside MedViT through ECB or LTB)
            x = self.model.features[0].patch_embed(x)  # Use the first PatchEmbed layer in features
            cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = self.model.pos_drop(x + self.model.pos_embed)

            # Get the attention weights from the last block
            last_block = self.model.blocks[-1]
            x = last_block.norm1(x)
            B, N, C = x.shape
            qkv = last_block.attn.qkv(x).reshape(B, N, 3, last_block.attn.num_heads,
                                                 C // last_block.attn.num_heads).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) * last_block.attn.scale
            attn = attn.softmax(dim=-1)  # B, num_heads, N, N

            # Average attention weights across heads
            attn = attn.mean(dim=1)  # B, N, N

            return attn

    def attention_heatmap(self, images, save_path=None):
        """Visualize attention heatmaps for given images"""
        images = images.to(self.device)
        batch_size = images.shape[0]
        attention_weights = self.get_attention_weights(images)

        # Calculate grid size based on patch size and image size
        patch_size = 16  # This should match your ViT patch_size
        img_size = images.shape[-1]  # Assuming square images
        grid_size = img_size // patch_size

        # Remove cls token and reshape
        attention_weights = attention_weights[:, 0, 1:].reshape(batch_size, grid_size, grid_size)

        fig, axs = plt.subplots(2, batch_size, figsize=(4 * batch_size, 8))

        for idx in range(batch_size):
            # Original image
            img = images[idx].cpu().permute(1, 2, 0)
            img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
            img = img.numpy()
            img = np.clip(img, 0, 1)

            axs[0, idx].imshow(img)
            axs[0, idx].axis('off')
            axs[0, idx].set_title('Original Image')

            # Attention heatmap
            sns.heatmap(attention_weights[idx].cpu(),
                        ax=axs[1, idx],
                        cmap='viridis',
                        cbar=idx == batch_size - 1)
            axs[1, idx].axis('off')
            axs[1, idx].set_title('Attention Map')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def visualize_features(self, loader, num_samples=1000, save_path=None):
        """Visualize feature embeddings using t-SNE"""
        features = []
        labels = []

        with torch.no_grad():
            for images, targets in loader:
                if len(features) >= num_samples:
                    break

                images = images.to(self.device)
                batch_features = self.model.forward_features(images)
                features.append(batch_features.cpu())
                labels.append(targets)

        features = torch.cat(features, dim=0)[:num_samples]
        labels = torch.tensor(labels, dtype=torch.int64)[:num_samples]

        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(features.numpy())

        # Plot
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels,
                              cmap='tab10', alpha=0.6)
        plt.colorbar(scatter)
        plt.title('t-SNE Visualization of Features')

        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_confusion_matrix(self, loader, save_path=None):
        """Plot confusion matrix for model predictions"""
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(12, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')

        if save_path:
            plt.savefig(save_path)
        plt.show()


def run_advanced_visualization(model, train_loader, val_loader, device, class_names):
    """Run all visualizations"""
    vis_tool = ViTVisualizer(model, device, class_names)

    # Get sample images
    sample_images, _ = next(iter(val_loader))

    # Generate and save visualizations
    vis_tool.attention_heatmap(sample_images[:4], 'attention_heatmap.png')
    vis_tool.visualize_features(val_loader, num_samples=1000, save_path='feature_tsne.png')
    vis_tool.plot_confusion_matrix(val_loader, save_path='confusion_matrix.png')