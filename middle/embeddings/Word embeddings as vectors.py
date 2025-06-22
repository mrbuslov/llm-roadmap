# pip install scikit-learn gensim
import gensim
from gensim.models import Word2Vec
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Sample corpus - in practice, you'd use a much larger dataset
sentences = [
    ['machine', 'learning', 'is', 'fun'],
    ['deep', 'learning', 'uses', 'neural', 'networks'],
    ['python', 'is', 'great', 'for', 'machine', 'learning'],
    ['neural', 'networks', 'are', 'powerful'],
    ['artificial', 'intelligence', 'will', 'change', 'the', 'world'],
    ['programming', 'with', 'python', 'is', 'enjoyable'],
    ['data', 'science', 'requires', 'machine', 'learning'],
    ['algorithms', 'solve', 'complex', 'problems'],
    ['computer', 'vision', 'uses', 'deep', 'learning'],
    ['natural', 'language', 'processing', 'is', 'challenging']
]

"""
Defines the number of dimensions in the vector representation of each word
Value options:

50-100: For small corpus or rapid prototyping
100-200: Standard choice for most tasks
200-300: For large corpus and complex semantic relationships
300+: For very large corpus (millions of documents)

✅ More dimensions = richer representation of semantics
❌ More dimensions = more memory and training time
❌ Too many dimensions can lead to overtraining
"""
VECTOR_SIZE = 100

"""
Determines how many words to the left and right of the target word are considered in training (for word semantic meaning)

window=1-3: To identify syntactic relations (parts of speech)
window=5-10: For semantic relations (meaning of words)
window=10+: For thematic relations (general topics)

Skip-gram vs CBOW:

Skip-gram: works better with larger windows (5-10)
CBOW: more effective with smaller windows (5-15)
"""
WINDOW_SIZE = 5

"""
Defines learning algorithm:
CBOW (Continuous Bag of Words) - sg=0
Principle: Predicts target word from context
Context: [learning, ?, training] → Prediction: "machine"
Benefits of CBOW:

✅ Faster learning
✅ Better for frequent words
✅ Requires less memory
✅ More stable vectors for frequent words

When to use CBOW:

Large text corpus
Limited computational resources
Focus on frequent words

Skip-gram - sg=1
Principle: Predicts context from target word
Target word: "machine" → Prediction: [learn, learning]
Benefits of Skip-gram:

✅ Better for rare words
✅ More accurate vectors for small corpus
✅ Better identifies semantic relations
✅ Works well with subsamples

When to use Skip-gram:

Small to medium-sized corpus
Many rare but important words
Need high precision of semantic relations
"""
ARCH_ALORITHM = 0

# Train Word2Vec model
print("Training Word2Vec model...")
model = Word2Vec(
    sentences,          # Input data for model training. The words must already be preprocessed (tokenized)
    vector_size=VECTOR_SIZE,    # dimensionality of word vectors. 
    window=WINDOW_SIZE,           # context window size. 
    min_count=1,        # ignore words with frequency less than this. Removes typos and random characters
    workers=4,          # number of threads. Important: More threads does not always mean faster due to synchronization overheads
    sg=0,              # 0 = CBOW, 1 = Skip-gram
    epochs=100         # number of training epochs. Determines how many times the algorithm traverses the entire corpus
)

# Save the model
model.save("word2vec.model")

# Load the model (optional - for demonstration)
# model = Word2Vec.load("word2vec.model")

print("\n=== Word2Vec Model Information ===")
print(f"Vocabulary size: {len(model.wv.key_to_index)}")
print(f"Vector size: {model.vector_size}")

# Get word vector
word = "learning"
if word in model.wv:
    vector = model.wv[word]
    print(f"\nVector for '{word}' (first 10 dimensions):")
    print(vector[:10])

# Find similar words
print("\n=== Similar Words ===")
try:
    similar_words = model.wv.most_similar('learning', topn=5)
    print(f"Words similar to 'learning':")
    for word, similarity in similar_words:
        print(f"  {word}: {similarity:.4f}")
except KeyError:
    print("Word not in vocabulary")

# Word analogy (king - man + woman = queen)
"""
Classic example: King - Man + Woman = Queen
The math behind analogies
python# Vector arithmetic
vector_result = vector_king - vector_man + vector_woman
# The result should be close to vector_queen
Concept Visualization
In vector space:
В векторном пространстве:
KING -----> MAN      (vector "masculinity")
 |           |
 |           |       (parallel vectors])
 ↓           ↓
QUEEN ----> WOMAN (same vector "masculinity")

"""
print("\n=== Word Analogies ===")
try:
    # This might not work well with our small corpus
    result = model.wv.most_similar(positive=['python', 'fun'], negative=['programming'], topn=3)
    print("python + fun - programming =")
    for word, similarity in result:
        print(f"  {word}: {similarity:.4f}")
except:
    print("Analogy couldn't be computed with this small corpus")

# Calculate similarity between words
print("\n=== Word Similarities ===")
word_pairs = [('machine', 'learning'), ('python', 'programming'), ('deep', 'neural')]
for word1, word2 in word_pairs:
    try:
        similarity = model.wv.similarity(word1, word2)
        print(f"Similarity between '{word1}' and '{word2}': {similarity:.4f}")
    except KeyError as e:
        print(f"One of the words not in vocabulary: {e}")

# Visualize word embeddings using PCA
print("\n=== Visualizing Word Embeddings ===")
def plot_embeddings(model, words=None):
    if words is None:
        words = list(model.wv.key_to_index.keys())[:20]  # Plot first 20 words
    
    # Get vectors for the words
    vectors = []
    labels = []
    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])
            labels.append(word)
    
    # Reduce dimensions to 2D using PCA
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1])
    
    # Add labels
    for i, label in enumerate(labels):
        plt.annotate(label, (vectors_2d[i, 0], vectors_2d[i, 1]), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.title('Word2Vec Embeddings Visualization (PCA)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.grid(True, alpha=0.3)
    plt.show()

# Plot the embeddings
plot_embeddings(model)

# Advanced: Working with larger datasets
print("\n=== Tips for Larger Datasets ===")
print("""
For real-world applications:

1. Use larger corpus (millions of sentences)
2. Preprocess text: lowercase, remove punctuation, tokenize
3. Filter rare and common words
4. Experiment with hyperparameters:
   - vector_size: 100-300 typical
   - window: 5-10 for Skip-gram, 5-15 for CBOW
   - min_count: 5-10 for filtering rare words
   - sg: 1 for Skip-gram (better for rare words)
        0 for CBOW (faster, better for frequent words)

Example preprocessing:
""")

# Example of text preprocessing
def preprocess_text(text):
    import re
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Split into words
    words = text.split()
    return words

sample_text = "Machine Learning is Amazing! It uses algorithms to solve problems."
processed = preprocess_text(sample_text)
print(f"Original: {sample_text}")
print(f"Processed: {processed}")

# Example: Loading pre-trained models
print("\n=== Using Pre-trained Models ===")
print("""
You can use pre-trained models like Google's Word2Vec:

# Download and load Google's pre-trained vectors
import gensim.downloader as api
model = api.load("word2vec-google-news-300")

# Or use other pre-trained models
# model = api.load("glove-wiki-gigaword-100")
# model = api.load("fasttext-wiki-news-subwords-300")
""")
