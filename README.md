analyzeSS.py updated to use Google Cloud Vision for OCR number extraction.

Setup
-----
1. Create a virtualenv and activate it (optional but recommended):

   python3 -m venv .venv
   source .venv/bin/activate

2. Install dependencies:

   pip install -r requirements.txt

3. Configure Google Cloud credentials (choose one):

   - Service account JSON (recommended for scripts):
       export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account.json

   - OR use ADC via gcloud:
       gcloud auth application-default login

Usage
-----
- `analyzeSS.analyze_screenshot(img)` accepts a PIL Image and returns a dict with keys:
  - raw_text: OCR'd full text (string)
  - numbers: list of numeric tokens (strings)
  - parsed_numbers: list of parsed numeric values (ints/floats or None)
  - count: number of numeric tokens found
  - error: present if an error occurred

- Example (from project root):

   python -c "from PIL import Image; import analyzeSS; img=Image.open('screenshot.png'); print(analyzeSS.analyze_screenshot(img))"

Notes
-----
- The Google Cloud Vision API will incur costs beyond the free tier for heavy usage. See Google Cloud pricing.
- The module with fail gracefully if credentials or packages are missing and returns an `error` field.

