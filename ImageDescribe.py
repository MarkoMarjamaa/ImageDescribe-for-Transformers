import os
import time
import threading
from typing import Dict, Optional

import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth
from urllib.parse import urlparse, urlunparse

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image
from io import BytesIO

import torch
from transformers import AutoModel, AutoProcessor

from requests_testadapter import Resp
from typing import Literal, TypedDict, Optional

class LocalFileAdapter(requests.adapters.HTTPAdapter):
	def build_response_from_file(self, request):
		file_path = request.url[7:]  # strip "file://"
		if not os.path.isfile(file_path):
			raise FileNotFoundError(f"File not found: {file_path}")

		with open(file_path, "rb") as file:
			buff = bytearray(os.path.getsize(file_path))
			file.readinto(buff)
			resp = Resp(buff)
			r = self.build_response(request, resp)
			return r

	def send(
		self,
		request,
		stream=False,
		timeout=None,
		verify=True,
		cert=None,
		proxies=None,
	):
		return self.build_response_from_file(request)


# ---- Configuration ----
MODEL_PATH = os.environ.get("R4B_MODEL_PATH", "./R-4B")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Allowlist of camera names -> image URL.
AuthType = Literal["none", "basic", "digest"]

class CameraConfig(TypedDict):
	url: str
	auth_type: AuthType
	username: Optional[str]
	password: Optional[str]
	verify_tls: bool

SECURITY_CAMERAS: dict[str, CameraConfig] = {
	"ExampleCamera": {
		"url": "http://server/directory/picture",
		"auth_type": "digest",   # try "basic" first; many cameras need "digest"
		"username": "admin",
		"password": "admin",
		"verify_tls": True,
	},
	"BlackbirdNest": {
		"url": "http://192.168.1.1/snap.jpeg",
		"auth_type": "none",   # try "basic" first; many cameras need "digest"
		"username": None,
		"password": None,
		"verify_tls": True,
	},
}

# Basic limits to avoid abuse / accidental overload
MAX_PROMPT_CHARS = int(os.environ.get("MAX_PROMPT_CHARS", "2000"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "2048"))  # adjust as needed

# ---- Global model state (loaded once) ----
model = None
processor = None

# A lock to serialize GPU generations if you want safety under concurrency.
# (Transformers generate on GPU can behave poorly under concurrent calls if VRAM is tight.)
gpu_lock = threading.Lock()

def load_model():
	global model, processor
	if model is not None and processor is not None:
		return

	# Load model
	m = AutoModel.from_pretrained(
		MODEL_PATH,
		torch_dtype=torch.float32,
		local_files_only=True,
		trust_remote_code=True,
	).to(DEVICE)

	# Load processor
	p = AutoProcessor.from_pretrained(
		MODEL_PATH,
		local_files_only=True,
		trust_remote_code=True,
	)

	model = m
	processor = p

def _strip_userinfo(url: str) -> str:
	"""
	Remove user:pass@ from url if present, to avoid leaking creds and parsing issues.
	"""
	p = urlparse(url)
	if p.username or p.password:
		netloc = p.hostname or ""
		if p.port:
			netloc += f":{p.port}"
		return urlunparse((p.scheme, netloc, p.path, p.params, p.query, p.fragment))
	return url

def fetch_image_from_camera(cam: CameraConfig) -> Image.Image:
	session = requests.session()
	session.mount("file://", LocalFileAdapter())

	url = cam["url"]


	try:
		if ( url.startswith("file://")):
			resp = session.get(url, stream=True)
		else:
			safe_url = _strip_userinfo(url)
			auth = None
			if cam.get("auth_type") == "basic":
				auth = HTTPBasicAuth(cam.get("username") or "", cam.get("password") or "")
			elif cam.get("auth_type") == "digest":
				auth = HTTPDigestAuth(cam.get("username") or "", cam.get("password") or "")
			headers = {
				"User-Agent": "camera-vision-service/1.0",
				"Accept": "image/*,*/*;q=0.8",
			}

			resp = session.get(
				safe_url,
				stream=True,
				timeout=10,
				auth=auth,
				headers=headers,
				verify=cam.get("verify_tls", True),
				allow_redirects=True,
			)

			resp.raise_for_status()

			ctype = (resp.headers.get("Content-Type") or "").lower()
			if "image" not in ctype:
				raise HTTPException(
					status_code=400,
					detail=f"Camera did not return an image. Content-Type={ctype} Status={resp.status_code}",
				)

		return Image.open(resp.raw).convert("RGB")

	except requests.HTTPError as e:
		www = ""
		try:
			www = resp.headers.get("WWW-Authenticate", "")
		except Exception:
			pass
		raise HTTPException(status_code=401, detail=f"HTTP error fetching camera image: {e}. WWW-Authenticate={www}")
	except Exception as e:
		raise HTTPException(status_code=400, detail=f"Failed to fetch/parse camera image: {e}")

def image_describe(camera: dict, prompt: str) -> str:
	"""
	Your ImageDescribe logic, but using globals model/processor and a safer fetch path.
	"""
	if processor is None or model is None:
		raise HTTPException(status_code=500, detail="Model not loaded")

	messages = [
		{
			"role": "user",
			"content": [
				{"type": "image"},
				{"type": "text", "text": prompt},
			],
		}
	]

	text = processor.apply_chat_template(
		messages,
		tokenize=False,
		add_generation_prompt=True,
		thinking_mode="auto",
	)

	image = fetch_image_from_camera(camera)

	inputs = processor(
		images=image,
		text=text,
		return_tensors="pt",
	).to(DEVICE)

	# Serialize GPU inference if desired
	with gpu_lock:
		generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

	output_ids = generated_ids[0][len(inputs.input_ids[0]) :]
	output_text = processor.decode(
		output_ids,
		skip_special_tokens=True,
		clean_up_tokenization_spaces=False,
	)
	return output_text.split('</think>\n\n')[-1]

# ---- FastAPI app ----
app = FastAPI(title="Camera Vision Service", version="1.0")

class AnalyzeRequest(BaseModel):
	camera_name: str = Field(..., description="Name of the security camera, e.g., FrontDoor")
	prompt: str = Field(..., description="Instruction for the vision model")

class AnalyzeResponse(BaseModel):
	camera_name: str
	prompt: str
	output: str
	latency_ms: int

@app.on_event("startup")
def on_startup():
	load_model()

@app.get("/health")
def health():
	return {
		"status": "ok",
		"device": DEVICE,
		"model_path": MODEL_PATH,
		"cameras": sorted(list(SECURITY_CAMERAS.keys())),
	}

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
	camera_name = req.camera_name.strip()
	prompt = req.prompt.strip()

	if not camera_name:
		raise HTTPException(status_code=400, detail="camera_name is required")
	if camera_name not in SECURITY_CAMERAS:
		raise HTTPException(
			status_code=404,
			detail=f"Unknown camera_name '{camera_name}'. Allowed: {sorted(list(SECURITY_CAMERAS.keys()))}",
		)

	if not prompt:
		raise HTTPException(status_code=400, detail="prompt is required")
	if len(prompt) > MAX_PROMPT_CHARS:
		raise HTTPException(status_code=400, detail=f"prompt too long (>{MAX_PROMPT_CHARS} chars)")

	t0 = time.time()
	output = image_describe(camera=SECURITY_CAMERAS[camera_name], prompt=prompt)
	latency_ms = int((time.time() - t0) * 1000)

	return AnalyzeResponse(
		camera_name=camera_name,
		prompt=prompt,
		output=output,
		latency_ms=latency_ms,
	)

if __name__ == "__main__":
	import uvicorn

	uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("IMAGEDESCRIBE_PORT", "8100")))

