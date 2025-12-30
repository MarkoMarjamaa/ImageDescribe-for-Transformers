# ImageDescribe-for-Transformers
Simple net service that runs transformers LLM with images, http:// or file:// from internal list. 

I created this because needed some interface for R-4B model, and it seemed there wont be quants for that 
In Open VLM Leaderboard R-4B was doing great for such a small model. 
https://huggingface.co/spaces/opencompass/open_vlm_leaderboard

Net service can be used from LLM as a tool when LLM don't have to know the url/username/password for cameras. This script contains array of cameras in use. 

# Downloading the R-4B
```
from huggingface_hub import snapshot_download
snapshot_download(repo_id="YannQi/R-4B",local_dir="R-4B")
```

# Installing packages
I'm using Ryzen 395+ so torch is for that
With Amd's version of "cuda" uses torch_dtype=torch.float32, Nvidia torch_dtype=torch.float16
```
pip install requests
pip install pillow

pip install ./triton-3.2.0+rocm7.1.0.git20943800-cp312-cp312-linux_x86_64.whl
pip install ./torch-2.6.0+rocm7.1.0.lw.git78f6ff78-cp312-cp312-linux_x86_64.whl
pip install ./torchaudio-2.6.0+rocm7.1.0.gitd8831425-cp312-cp312-linux_x86_64.whl
pip install ./torchvision-0.21.0+rocm7.1.0.git4040d51f-cp312-cp312-linux_x86_64.whl

pip install transformers
pip install requests_testadapter

pip install fastapi
pip install uvicorn
```

# Using the service

## Health check
```
curl http://localhost:8100/health
```

## Test
```
curl -X POST http://localhost:8100/analyze \
  -H "Content-Type: application/json" \
  -d '{"camera_name":"BlackBirdNest","prompt":"Describe the scene and is the blackbird in it's nest."}'
```

## Tool definition
```
	tools = [
	{
		"type": "function",
		"function": {
			"name": "analyze_security_camera",
			"description": "Analyze a snapshot from a named security camera using the vision service.",
			"parameters": {
				"type": "object",
				"properties": {
					"camera_name": {
						"type": "string",
						"description": "One of: ExampleCamera, BlackbirdNest"
						},
					"prompt": { "type": "string", "description": "What to look for in the image." }
				},
				"required": ["camera_name", "prompt"]
			}
		},
	},
]
```
## Example of tool using
```
	tool_calls = message.get("tool_calls", [])
	for tool_call in tool_calls:
		name = tool_call["function"]["name"]
		args_str = tool_call["function"]["arguments"] or "{}"
		try:
			args = json.loads(args_str)
		except json.JSONDecodeError:
			args = {}
		
		if name == "analyze_security_camera":
			args.get("camera_name")
			args.get("prompt")

			url = f"http://localhost:8100/analyze"
			headers = {"Content-Type": "application/json"}

			payload = {
				"camera_name": args.get("camera_name"),
				"prompt": args.get("prompt"),
			}

			resp = requests.post(url, headers=headers, json=payload, timeout=40000)
			resp.raise_for_status()
			print(str(resp.json()["output"]))
```

## Service definition
This service runs in virtual-env. 
```
[Unit]
Description=ImageDescribe(R-4B)
After=network.target
[Service]
User=marko

WorkingDirectory=/home/mysecretusername/ImageDescribe
ExecStart=/home/mysecretusername/ImageDescribe/bin/python /home/mysecretusername/ImageDescribe/ImageDescribe.py
Restart=always
[Install]
WantedBy=multi-user.target
```

