import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# diameter classes
diameters = [2, 4, 7, 11]


# -----------------------------
# 1. Generate synthetic patches
# -----------------------------
def generate_circle_patch(diameter):
    patch = np.zeros((16, 16), dtype=np.float32)
    radius = diameter / 2.0

    # random center fully inside patch with enough margin for the circle
    cx = np.random.uniform(radius, 16.0 - radius)
    cy = np.random.uniform(radius, 16.0 - radius)

    # approximate pixel coverage with a 4x4 subpixel grid
    sub = 4
    inv_sub2 = 1.0 / (sub * sub)
    offsets = (np.arange(sub) + 0.5) / sub

    for y in range(16):
        for x in range(16):
            inside = 0
            for oy in offsets:
                for ox in offsets:
                    px = x + ox
                    py = y + oy
                    if (px - cx) ** 2 + (py - cy) ** 2 <= radius ** 2:
                        inside += 1
            patch[y, x] = inside * inv_sub2 * 255.0

    return patch.flatten() / 255.0  # normalize


def create_dataset(n_samples=5000):
    X = []
    y = []
    global diameters

    for _ in range(n_samples):
        d = np.random.choice(diameters)
        patch = generate_circle_patch(d)
        X.append(patch)
        y.append(diameters.index(d))  # class index

    return torch.tensor(np.array(X), dtype=torch.float32), \
           torch.tensor(np.array(y), dtype=torch.long)


# -----------------------------
# 2. Define model
# -----------------------------
class PatchEmbeddingModel(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        # This linear layer IS the W matrix
        self.W = nn.Linear(256, embed_dim)

        # classifier head
        global diameters
        self.classifier = nn.Linear(embed_dim, len(diameters))

    def forward(self, x):
        embedding = self.W(x)          # semantic embedding
        embedding = torch.relu(embedding)
        out = self.classifier(embedding)
        return out, embedding


# -----------------------------
# 3. Train model
# -----------------------------
model = PatchEmbeddingModel(embed_dim=32)
checkpoint_path = "model.pt"

if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Loaded previously saved model from {checkpoint_path}")
else:
    print(f"No model found at '{checkpoint_path}', so generate one..")
    print("Generating dataset and training model...")
    X, y = create_dataset(16000)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training model...")
    for epoch in range(200):
        optimizer.zero_grad()
        out, emb = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss = {loss.item():.4f}")

    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved model checkpoint to {checkpoint_path}")

# -----------------------------
# 4. Show learned W matrix
# -----------------------------
print("\nLearned W matrix (shape):", model.W.weight.shape)
# print(model.W.weight)

key_pressed = {'space': False}

# Wait for space key press
def on_key(event):
    if event.key == ' ':
        key_pressed['space'] = True

# Create figure once at the start
fig, (ax1) = plt.subplots(1, 1, figsize=(4, 4))
fig.canvas.mpl_connect('key_press_event', on_key)

model.eval()

for i in range(15):
    key_pressed['space'] = False
    
    random_diameter = np.random.choice(diameters)
    random_patch = generate_circle_patch(random_diameter)
    random_patch_tensor = torch.tensor(random_patch, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output, _ = model(random_patch_tensor)
        predicted_class = output.argmax(dim=1).item()

    predicted_diameter = diameters[predicted_class]
    is_correct = predicted_diameter == random_diameter

    print(f"{i}: Random patch diameter: {random_diameter}")
    print(f"Model predicted class index: {predicted_class}, diameter: {predicted_diameter}, result={is_correct}")

    # Clear and update subplots
    ax1.clear()
    
    # Original patch
    ax1.imshow(random_patch.reshape(16, 16), cmap='gray')
    ax1.set_title(f"Original: diameter={random_diameter}")
    ax1.axis('off')
    
    plt.suptitle(f"Iteration {i} - Press SPACE to continue")
    plt.draw()

    # Wait for space key press
    while not key_pressed['space']:
        plt.pause(0.1)

plt.close()
