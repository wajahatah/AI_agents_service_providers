from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os

def run_inference(model_name: str, questions: list, max_new_tokens: int = 2048, output_dir="outputs"):
    """
    Load a HuggingFace model and run inference on a list of questions.
    """
    print(f"\nLoading model: {model_name} ...")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",       # automatically use GPU if available
        torch_dtype="auto"       # pick best dtype automatically
    )

    max_model_tokens = getattr(model.config, "max_position_embeddings", 2048)  # fallback
    print(f"🔧 Model max token limit: {max_model_tokens}")


    # Build pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{model_name.replace('/', '_')}.txt")

    with open(file_path, "a", encoding="utf-8") as f:
        f.write("\n" + "="*80 + "\n")
        f.write(f"Responses from model: {model_name}\n")
        f.write("="*80 + "\n\n")

        # Loop through questions
        for i, question in enumerate(questions, 1):
            print(f"\n{'='*30} Question {i} {'='*30}\n")
            print(f"🟢 {question}\n")

            # if rag == True:
            #     prompt = prompt = f"""
            #         You are given the following context:

            #         {context}

            #         Answer the question using ONLY this context.
            #         If the answer is not in the context, say "I don't know."

            #         Question: {question}
            #             """

            prompt_tokens = len(tokenizer.encode(question))
            max_new_tokens = max_model_tokens - prompt_tokens - 10

            if max_new_tokens < 100:  # fallback
                max_new_tokens = 100

            response = generator(
                question,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.7
            )

            print("🔵 Model Response:\n")
            print(response[0]["generated_text"])
            print("\n" + "="*80 + "\n")

            answer = response[0]["generated_text"]
            f.write(f"Q{i}: {question}\n")
            f.write(f"Answer: {answer}\n")
            f.write("-"*60 + "\n")



# ---------------- Example usage ----------------
if __name__ == "__main__":
    # 🔧 Choose the HuggingFace model name
    model_name = "Qwen/Qwen3-0.6B"   # Exampledeepseek-ai/deepseek-coder-1.3b-instruct

    rag = False

    # ✍️ Paste all your questions here
    questions = [
        "I want to fly from Karachi to Istanbul next Thursday evening. Provide me the available flights, their schedule, any connecting flight, and the price. Book the flight after taking my choices",
        "Book me a dentist appointment in Gulistan-e-Johor, Karachi, provide me the list of available dentist on saturday evening, what are their reviews, the doctor fees and any other related stuff.",
        "I have a connectingn flight from Karachi to Toronto with a connecting flight from Doha with a layover of 8 hours, suggest me some good places which I can visit during the time, book the cab for those places and do reservation where needed.",
        "Book me a hotel in morroco which is close to the park and good resturants, and must have a pool and gym. Also make a reservation in the resturant for 2 peoples."
        # "I run a resturant in the long island, I have all the pakistani and indian foods in the resturant and delivery time will be 30 mins any where in the long island. I also deliver in the new york and new jersy but the deliver time is a bit long, around 60 to 90 mins. Dine is also available."
    ]

    run_inference(model_name, questions )
