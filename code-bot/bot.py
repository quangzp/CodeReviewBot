import ollama

MODEL_NAME = "deepseek-coder:6.7b"

def review_code_instruct(file_content, filename):

    prompt = f"""<\uff5cbegin\u2581of\u2581sentence\uff5c>You are an expert in coding and code peer-reviewing.### Instruction:\n### Code review comment generation\nGenerate a review comment that you consider perfect for the given code changes.\nA review comment should highlight the main issues, improvements, or suggestions for the code changes.\nThe generated review comment should be concise, relevant, clear, useful, and complete.\n\n### Code changes:\n@@ -827,6 +827,10 @@ void initOptions()\n                                    _(\"Adjust the volume of the music being played in the background.\"),\n                                    0, 200, 100, COPT_CURSES_HIDE\n                                   );\n+    OPTIONS[\"SOUND_VOLUME\"] = cOpt(\"graphics\", _(\"Sound Volume\"),\n+                                   _(\"Adjust the volume of sound effects being played by the game.\"),\n\n### Response:\n"""

    try:
        response = ollama.generate(
            model=MODEL_NAME,
            prompt=prompt,
            options={
                "temperature": 0.1, # Để thấp để model tập trung, ít sáng tạo linh tinh
                "num_ctx": 4096
            },
            stream=False
        )
        return response['response']
    except Exception as e:
        return f"Error: {str(e)}"

# Test
code_to_test = """
def connect_db(password):
    query = "SELECT * FROM users WHERE pass = '" + password + "'"
    execute(query)
"""

print(review_code_instruct(code_to_test, "db.py"))