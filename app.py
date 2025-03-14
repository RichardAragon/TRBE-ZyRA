import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

# Load Sentence-BERT model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Get embeddings for four phrases
phrases = ["Hello World", "Good Morning", "How Are You", "See You Later"]
embeddings = [normalize([sentence_model.encode([p])[0]])[0] for p in phrases]

# Set parameters
dim = 384  # Sentence-BERT embedding dimension
grid_size = 512  # Full grid size
quarter_grid = grid_size // 2  # 256x256 quadrants

# Create four independent projection matrices
np.random.seed(42)
projection_matrices = [normalize(np.random.randn(grid_size * gridsize, dim), axis=0) for  in range(4)]

# Encode each phrase into full grids
semantic_fields = [proj @ emb for proj, emb in zip(projection_matrices, embeddings)]
semantic_fields = [field.reshape(grid_size, grid_size) for field in semantic_fields]

# Multiplex into 2x2 grid
semantic_field_phrase = np.zeros((grid_size, grid_size))
semantic_field_phrase[:quarter_grid, :quarter_grid] = semantic_fields[0][:quarter_grid, :quarter_grid]  # TL
semantic_field_phrase[:quarter_grid, quarter_grid:] = semantic_fields[1][:quarter_grid, quarter_grid:]  # TR
semantic_field_phrase[quarter_grid:, :quarter_grid] = semantic_fields[2][quarter_grid:, :quarter_grid]  # BL
semantic_field_phrase[quarter_grid:, quarter_grid:] = semantic_fields[3][quarter_grid:, quarter_grid:]  # BR

# Apply simple modulation
zyra_beam_phrase = semantic_field_phrase * 0.7

# Visualize the multiplexed beam
plt.figure(figsize=(10, 6))
plt.imshow(zyra_beam_phrase, cmap='plasma', extent=[0, grid_size, 0, grid_size])
plt.axvline(x=quarter_grid, color='white', linestyle='--', label='Vertical Split')
plt.axhline(y=quarter_grid, color='white', linestyle='--', label='Horizontal Split')
plt.title("TRBE-ZyRA 2x2 Multiplexed Beam\nTL: 'Hello World', TR: 'Good Morning'\nBL: 'How Are You', BR: 'See You Later'")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.colorbar(label='Field Intensity')
plt.legend()
plt.show()

# Decode each quadrant
pseudo_inverses = [np.linalg.pinv(proj) for proj in projection_matrices]
beam_flat = zyra_beam_phrase.flatten()

# Reconstruct full beams for each quadrant with corrected indexing
decoded_embeddings = []
for i in range(4):
    beam_reconstructed = np.zeros(grid_size * grid_size)
    row, col = divmod(i, 2)
    # Calculate the starting index in the flattened beam for this quadrant
    quad_start = (row * quarter_grid * grid_size) + (col * quarter_grid)
    for r in range(quarter_grid):
        # Source: row in the flattened beam
        src_idx = quad_start + (r * grid_size)
        # Destination: corresponding row in the full grid
        dst_idx = (r + row * quarter_grid) * grid_size + (col * quarter_grid)
        beam_reconstructed[dst_idx:dst_idx + quarter_grid] = beam_flat[src_idx:src_idx + quarter_grid]
    decoded_emb = pseudo_inverses[i] @ beam_reconstructed
    decoded_embeddings.append(normalize([decoded_emb])[0])

# Calculate similarities
similarities = [cosine_similarity([emb], [dec])[0][0] for emb, dec in zip(embeddings, decoded_embeddings)]

# Print similarities
for phrase, sim in zip(phrases, similarities):
    print(f"Similarity for '{phrase}': {sim:.4f}")

# Plot comparisons
plt.figure(figsize=(12, 10))
for i, (phrase, emb, dec, sim) in enumerate(zip(phrases, embeddings, decoded_embeddings, similarities)):
    plt.subplot(2, 2, i + 1)
    plt.plot(emb, label=f"Original '{phrase}'")
    plt.plot(dec, label=f"Decoded '{phrase}'", linestyle='--')
    plt.title(f"'{phrase}' | Similarity: {sim:.4f}")
    plt.legend()
plt.tight_layout()
plt.show()
