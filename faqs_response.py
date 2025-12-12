import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer, util

# ==============================
# CONFIGURATION
# ==============================
MODEL_PATH = "mistralai/Mistral-7B-Instruct-v0.2"#"ibm-granite/granite-4.0-micro"#"Qwen/Qwen3-0.6B"   # your downloaded model path or ID
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7

pdf = "stiching"

# Context (brief project description)
Stiching_title = "stiching"
stiching_context = """
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
qa_data_stiching = [
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
    },
    {
        "id": 4,
        "question": "Why is the sewing machine making a loud noise?",
        "expected_answer": "A loud noise may indicate that the machine needs cleaning or oiling. Check for any lint or thread caught in the bobbin area. If necessary, refer to the manual for instructions on how to lubricate the machine."
    },
    {
        "id": 5,
        "question": "How do I change the presser foot on the BEL0197?",
        "expected_answer": "To change the presser foot, raise the needle and presser foot lever. Release the current foot by pressing the lever at the back of the foot holder. Position the new foot under the holder and lower the presser foot lever to snap it into place."
    },
    {
        "id": 6,
        "question": "What steps should I follow to clean the sewing machine?",
        "expected_answer": "Turn off the machine and unplug it. Remove the needle, presser foot, and bobbin. Using a small brush, gently clean out any lint or dust from the bobbin area and feed dogs. Wipe the exterior with a soft cloth."
    },
    {
        "id": 7,
        "question": "How can I fix the bobbin winding issue?",
        "expected_answer": "Ensure the bobbin is placed correctly on the winder spindle and that the thread is wrapped around the bobbin tension guide. Start the machine and make sure the bobbin is winding evenly. If not, check for any tangles or misalignment."
    },
    {
        "id": 8,
        "question": "Why does the fabric bunch up under the needle?",
        "expected_answer": "Fabric bunching can result from incorrect tension settings or improper threading. Rethread the machine and check the tension dial settings. Also, ensure the needle is suitable for the fabric type."
    },
    {
        "id": 9,
        "question": "What causes the needle to break frequently?",
        "expected_answer": "Frequent needle breaks can be caused by pulling the fabric too hard, using the wrong needle size, or hitting a pin. Use the correct needle for your fabric and avoid sewing over pins."
    },
    {
        "id": 10,
        "question": "How do I select different stitch patterns on the BEL0197?",
        "expected_answer": "To select a stitch pattern, turn the stitch selector dial on the front of the machine. Align the desired stitch number with the indicator. Adjust the stitch length and width as needed for your project."
    }
]


stove_title = "stove"
stove_context = """
The Beldray BA488430 Large Electric Stove With Flame Effects is a versatile heating
appliance designed to combine functionality with aesthetic appeal. It features a robust
design that mimics a traditional wood-burning stove, providing a cozy ambiance with its
realistic flame effects. The unit operates using a sturdy electric heating element, which
offers efficient heat output suitable for medium to large-sized rooms. This stove is
equipped with adjustable heat settings, allowing users to easily control the temperature
to their preference. A key feature is the independent flame effect, which can be used with
or without heat, providing year-round ambiance. The stove is designed with a large
viewing window, offering a clear and immersive view of the flame effects. Its construction
includes durable materials, ensuring longevity and safety during operation. The
appliance is equipped with overheat protection, automatically shutting off if it becomes
too hot, enhancing safety for users. Installation is straightforward, requiring only a
standard electrical outlet, making it convenient for various living spaces without the need
for additional ventilation or chimney systems. The Beldray BA488430 is designed with
user-friendly controls that are intuitive and easy to operate, ensuring a seamless
experience. This electric stove not only serves as an efficient heating solution but also as
an elegant focal point in any room, combining modern technology with classic design
elements.
"""

qa_data_stove = [
        {
        "id": 1,
        "question": "How do I assemble the Beldray BA488430 Large Electric Stove?",
        "expected_answer": "To assemble the Beldray BA488430, follow the instructions provided in the user manual. Ensure all components are present and use the provided screws and tools for assembly. Secure each part firmly but avoid over-tightening."
    },
    {
        "id": 2,
        "question": "What should I do if the flame effects are not working?",
        "expected_answer": "If the flame effects are not working, check the power supply and ensure the unit is plugged in and switched on. Inspect the bulb and replace it if necessary. Consult the manual for further troubleshooting steps or contact customer support."
    },
    {
        "id": 3,
        "question": "Can I use the heater function without the flame effects?",
        "expected_answer": "Yes, the heater function can be used independently of the flame effects. Refer to the control panel instructions to activate the heater only mode."
    },
    {
        "id": 4,
        "question": "How do I clean and maintain the electric stove?",
        "expected_answer": "To clean the stove, ensure it is turned off and unplugged. Use a soft, damp cloth to wipe the exterior and avoid abrasive cleaners. Regularly check and clean the air vents to ensure efficient operation."
    },
    {
        "id": 5,
        "question": "What is the recommended way to store the electric stove when not in use?",
        "expected_answer": "Store the electric stove in a cool, dry place away from direct sunlight. Ensure it is unplugged and covered with a dust cover if possible to prevent dust accumulation."
    },
    {
        "id": 6,
        "question": "How can I improve the energy efficiency of the Beldray BA488430?",
        "expected_answer": "To improve energy efficiency, use the thermostat to control the temperature and only use the heater when necessary. Keep the room insulated to retain warmth."
    },
    {
        "id": 7,
        "question":  "What is the maximum heating capacity of the Beldray BA488430?",
        "expected_answer": "The Beldray BA488430 has a maximum heating capacity suitable for medium-sized rooms. Refer to the product specifications in the user manual for exact details."
    },
    {
        "id": 8,
        "question": "Is it safe to leave the electric stove unattended?",
        "expected_answer": "While the stove has safety features, it is not advisable to leave it unattended for extended periods. Always ensure safety precautions are followed when in use."
    },
    {
        "id": 9,
        "question":  "What should I do if the stove overheats?",
        "expected_answer": "If the stove overheats, it will automatically shut off. Allow it to cool down before resetting. Check for blockages in the vents and ensure the thermostat is working correctly."
    },
    {
        "id": 10,
        "question": "How can I replace the bulb for the flame effects?",
        "expected_answer": "To replace the bulb, ensure the stove is turned off and unplugged. Access the bulb compartment as per the manual instructions, remove the old bulb, and insert a new one compatible with the stove's specifications."
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
similarity_scores = []


qa_data = eval(f"qa_data_{pdf}")
context = eval(f"{pdf}_context")
title = pdf
# print(qa_data)

for qa in qa_data:
    q = qa["question"]
    expected = qa["expected_answer"]
    full_prompt = f"Context:\n{context}\n\nQuestion: {q}\nAnswer:"
    print("ques:", q)
    try:
        output = pipe(full_prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)[0]["generated_text"]
        # Extract only the answer part
        answer = output.split("Answer:")[-1].strip()
    except Exception as e:
        print(f"⚠️ Error generating answer for question: {q} — {e}")
        answer = ""

    similarity = compare_answers(answer, expected)
    print("similarity", similarity)
    similarity_scores.append(similarity)
    # results.append({
    #     "question": q,
    #     "expected_answer": expected,
    #     "model_answer": answer,
    #     "similarity_score": similarity
    # })
    result_block = f"""
    ==============================
    Question ID: {qa["id"]}
    Question: {q}

    Expected Answer:
    {expected}

    Model Answer:
    {answer}

    Similarity Score: {similarity:.3f}
    ==============================
    """
    results.append(result_block)

# ==============================
# SAVE RESULTS
# ==============================
# df = pd.DataFrame(results)
os.makedirs("results", exist_ok=True)
# title = f"{title} + {MODEL_PATH}"
model = f"{MODEL_PATH.replace('/', '_')}"
txt_path = os.path.join("results", f"{title}_{model}_evaluation.txt")
# os.makedirs(txt_path, exist_ok=True)

# csv_path = os.path.join("results", f"{MODEL_PATH.replace('/', '_')}_evaluation.csv")
# df.to_csv(csv_path, index=False)

# avg_score = df["similarity_score"].mean()
# print("\n======================")
# print(f"📊 Evaluation Summary for {MODEL_PATH}")
# print("======================")
# print(df[["question", "similarity_score"]])
# print(f"\nAverage similarity: {avg_score:.3f}")
# print(f"\n✅ Results saved at: {csv_path}")

with open(txt_path, "w", encoding="utf-8") as f:
    f.write(f"Evaluation Report for {MODEL_PATH}\n")
    f.write("=" * 60 + "\n")
    for block in results:
        f.write(block)
    f.write("=" * 60 + "\n")

print(f"\n✅ Evaluation complete! Results saved at: {txt_path}")

    