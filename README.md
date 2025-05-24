<h1>MultiWOZ-Based Task-Oriented Chatbot Using T5 Transformers</h1>
<h2>(Hotel, Restaurant, Taxi Domains)</h2>

<h2>📌 Introduction</h2>
<p>
Task-oriented chatbots are essential in automating service-based interactions in domains like hotel reservations, restaurant bookings, and taxi services. 
This project presents a chatbot trained on the MultiWOZ v2.2 dataset, leveraging the T5 transformer architecture for generating accurate and context-aware responses.
</p>

<h2>📂 Dataset</h2>
<p>
We use the MultiWOZ v2.2 dataset and filter it for the hotel, restaurant, and taxi domains. Only the training and testing subsets are used. 
The dataset contains multi-turn dialogues labeled with services and speaker roles.
</p>
<p>
MultiWOZ is a large-scale, human-human, task-oriented dialogue dataset that aligns perfectly with real-world customer service scenarios. 
Its dialogues simulate natural user behavior when interacting with service agents, making it especially valuable for training chatbots that must understand and respond appropriately in multi-turn interactions.
</p>

<h2>⚙️ Model Selection and Preprocessing</h2>

<h3>Initial Experiments with T5-Small</h3>
<p>
We began with <code>t5-small</code> (60M parameters) for quick experimentation. Preprocessing steps included tokenization, stopword removal, and lemmatization using NLTK.
</p>

<h3>Tokenization and Data Preparation</h3>
<p>
We used HuggingFace's <code>AutoTokenizer</code> to tokenize dialogues and padded/truncated them to 128 tokens. Data was structured using PyTorch's <code>Dataset</code> class.
</p>

<h3>Limitations of T5-Small</h3>
<ul>
  <li>Struggled with long-term context</li>
  <li>Weak in domain-specific slot-filling</li>
  <li>Generated partial or generic replies</li>
</ul>

<h3>Transition to T5-Base</h3>
<p>
We upgraded to <code>t5-base</code> (220M parameters) to improve coherence, belief state modeling, and generalization across domains. 
This came with increased training time and memory needs, but yielded much better results.
</p>

<h3>Model Comparison</h3>
<table border="1">
  <thead>
    <tr>
      <th>Aspect</th>
      <th>T5-small</th>
      <th>T5-base</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Model Size</td>
      <td>60M parameters</td>
      <td>220M parameters</td>
    </tr>
    <tr>
      <td>Training Speed</td>
      <td>Faster</td>
      <td>Slower</td>
    </tr>
    <tr>
      <td>Memory Usage</td>
      <td>Low</td>
      <td>High</td>
    </tr>
    <tr>
      <td>Output Quality</td>
      <td>Often generic</td>
      <td>Coherent and context-aware</td>
    </tr>
    <tr>
      <td>Domain Adaptability</td>
      <td>Weak</td>
      <td>Strong</td>
    </tr>
    <tr>
      <td>Use Case</td>
      <td>Early experiments</td>
      <td>Production-quality training</td>
    </tr>
  </tbody>
</table>

<h2>🧱 Custom Dataset Class</h2>
<p>
A PyTorch <code>Dataset</code> class was built to handle input-output pairs with attention masking. Padding tokens are ignored in loss computation.
</p>

<h2>🛠️ Training Setup</h2>
<p>
We used Hugging Face’s <code>Seq2SeqTrainer</code> with the following config:
</p>
<ul>
  <li>Epochs: 10</li>
  <li>Batch size: 5</li>
  <li>Eval steps: 500</li>
  <li>Learning rate: 1e-5</li>
  <li>Logging and checkpointing enabled</li>
</ul>
<p>
Training was done locally using GPU acceleration in PyCharm.
</p>

<h2>📊 Evaluation</h2>
<p>
We evaluated performance using BLEU (via NLTK) and ROUGE metrics. BLEU measured fluency and adequacy; ROUGE measured lexical overlap with references.
</p>

<h2>✅ Results</h2>
<p>
The model generated fluent, relevant, and domain-specific responses. BLEU and ROUGE scores showed strong performance in hotel, restaurant, and taxi domains.
</p>

<h2>🔮 Conclusion and Future Work</h2>
<p>
We successfully applied T5 transformers for task-oriented dialogue generation. Future improvements include:
</p>
<ul>
  <li>Expanding to more domains</li>
  <li>Adding dialogue state tracking</li>
  <li>Personalizing responses for users</li>
  <li>Human-based evaluation</li>
</ul>
