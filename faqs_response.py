import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer, util

# ==============================
# CONFIGURATION
# ==============================
MODEL_PATH = "Qwen/Qwen3-0.6B"   # your downloaded model path or ID
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7

# Context (brief project description)
context = """
The Beldray BEL0197 12 Stitch Sewing Machine is designed for both novice and
experienced users seeking versatility and ease of use. It features a robust build with a
compact, portable design, making it ideal for sewing enthusiasts with limited space. The
machine offers 12 distinct stitch patterns, enabling users to perform a range of sewing
tasks, from basic stitching to decorative designs. It is equipped with an easy-to-use dial
for selecting stitch patterns, allowing for quick adjustments to suit various fabric types
and sewing requirements.
The BEL0197 includes a four-step buttonhole function, simplifying the process of
creating neat and consistent buttonholes. Its adjustable stitch length and zigzag width
provide greater control over stitch customization. The machine is fitted with a powerful
motor that ensures steady performance, even when working with thicker fabrics. A builtin 
LED light illuminates the sewing area, enhancing visibility and precision during sewing
tasks.
Additional features include a thread cutter for convenience, a reverse sewing lever for
reinforcing stitches, and a free arm for sewing cylindrical items like sleeves and cuffs. The
Beldray BEL0197 also offers easy bobbin winding and a drop-in bobbin system for
straightforward threading. Accessories such as extra bobbins, needles, and a foot pedal
are included, providing users with everything needed to start sewing immediately. With
its user-friendly operation and comprehensive features, the Beldray BEL0197 is an
excellent choice for anyone looking to explore sewing projects with ease and efficiency
"""

# Questions and expected answers
qa_data = [
    {
        "id": 1,
        "question": "How do I thread the Beldray BEL0197 sewing machine?",
        "expected_answer": "To thread the machine, first raise the needle to its highest position. Place the spool of thread on the spool pin and guide the thread through the thread guide, down the tension dial, and into the needle. Ensure the thread is properly secured in the take-up lever"
    },
    {
        "id": 2,
        "question": "What should I do if the stitches are skipping?",
        "expected_answer": "Skipped stitches can be due to a dull or bent needle. Replace the needle with a new one, ensuring it is the correct type for your fabric. Also, check that the machine is threaded correctly."
    },
    {
        "id": 3,
        "question": "How can I adjust the tension on my sewing machine?",
        "expected_answer": "To adjust the tension, use the tension dial located on the front of the machine. For tighter tension, turn the dial to a higher number. For looser tension, turn it to a lower number. Always test on a scrap piece of fabric first."
    }
]

# Similarity model for automatic scoring
scorer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ==============================
# HELPER FUNCTIONS
# ==============================
def compare_answers(model_answer, reference_answer):
    emb1 = scorer.encode(model_answer, convert_to_tensor=True)
    emb2 = scorer.encode(reference_answer, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2).item())

# ==============================
# LOAD MODEL PIPELINE
# ==============================
print(f"\n🚀 Loading model: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# ==============================
# RUN EVALUATION
# ==============================
results = []

for qa in qa_data:
    q = qa["question"]
    expected = qa["expected_answer"]
    full_prompt = f"Context:\n{context}\n\nQuestion: {q}\nAnswer:"

    try:
        output = pipe(full_prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)[0]["generated_text"]
        # Extract only the answer part
        answer = output.split("Answer:")[-1].strip()
    except Exception as e:
        print(f"⚠️ Error generating answer for question: {q} — {e}")
        answer = ""

    similarity = compare_answers(answer, expected)
    results.append({
        "question": q,
        "expected_answer": expected,
        "model_answer": answer,
        "similarity_score": similarity
    })

# ==============================
# SAVE RESULTS
# ==============================
df = pd.DataFrame(results)
os.makedirs("results", exist_ok=True)
csv_path = os.path.join("results", f"{MODEL_PATH.replace('/', '_')}_evaluation.csv")
df.to_csv(csv_path, index=False)

avg_score = df["similarity_score"].mean()
print("\n======================")
print(f"📊 Evaluation Summary for {MODEL_PATH}")
print("======================")
print(df[["question", "similarity_score"]])
print(f"\nAverage similarity: {avg_score:.3f}")
print(f"\n✅ Results saved at: {csv_path}")
