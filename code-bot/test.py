from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "deepseek-ai/deepseek-coder-1.3b-instruct"
# 1. Nạp tokenizer và mô hình gốc
tokenizer = AutoTokenizer.from_pretrained("D:\\GENAI\\K1\\LLM\\results\\fine-tuned-model") # Dùng tệp tokenizer bạn vừa lưu
base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

# 2. Nạp "não bộ" LoRA vào mô hình gốc [cite: 291]
model = PeftModel.from_pretrained(base_model, "D:\\GENAI\\K1\\LLM\\results\\fine-tuned-model")

# 3. Chuẩn bị đoạn code cần review
prompt = "<\uff5cbegin\u2581of\u2581sentence\uff5c>You are an expert in coding and code peer-reviewing.### Instruction:\n### Code review comment generation\nGenerate a review comment that you consider perfect for the given code changes.\nA review comment should highlight the main issues, improvements, or suggestions for the code changes.\nThe generated review comment should be concise, relevant, clear, useful, and complete.\n\n### Code changes:\n@@ -473,17 +481,33 @@ func validateTopicSubscription(ts manifest.TopicSubscription, validTopicARNs []s\n \t}\n \n \t// Check that the topic is included in the list of available topics\n+\ttopicName := fmt.Sprintf(resourceNameFormat, app, env, ts.Service, ts.Name)\n \tfor _, topicARN := range validTopicARNs {\n-\t\tsplitArn := strings.Split(topicARN, \":\")\n-\t\ttopicName := strings.Split(splitArn[len(splitArn)-1], \"-\")\n-\t\tif len(topicName) < 4 {\n-\t\t\tcontinue\n+\t\tarn, err := arn.Parse(topicARN)\n+\t\tif err != nil {\n+\t\t\treturn err\n\n### Response:\n"

inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=128)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))