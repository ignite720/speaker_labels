import os
import time
from datetime import timedelta
import logging
import json
from pathlib import Path

from pyannote.audio import Pipeline
import torch
import gradio as gr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        #logging.FileHandler(filename=Path().home() / f"{input_filename_part0}-speaker_labels.txt", mode="w"),
    ],
)

hf_access_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
assert(hf_access_token)

def main(file):
    #foo = os.path.expandvars("$HOME/foo")

    input_file = Path(file.name).expanduser()
    logging.info(input_file)
    assert("-" in input_file.name and input_file.is_file())

    input_filename_part0 = input_file.name.split("-")[0]
    logging.info(input_filename_part0)

    t0 = time.time()
    logging.info(f"t0: {t0}")

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_access_token)
    pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    diarization = pipeline(input_file)

    data = {
        "speaker_labels": [],
        "speakers": None,
    }

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        label = {
            "start_millis": 0,
            "end_millis": 0,
            "speaker_label": speaker,
            "text": "",
            "start_secs": turn.start,
            "end_secs": turn.end,
            "start_text": str(timedelta(seconds=turn.start)),
            "end_text": str(timedelta(seconds=turn.end)),
        }
        data["speaker_labels"].append(label)
        logging.warning(f"{label}")
    data["speakers"] = sorted(set(item["speaker_label"] for item in data["speaker_labels"]))

    with open(Path().home() / f"{input_filename_part0}-speaker_labels.json", mode="w", encoding="utf-8", newline="\n") as fp:
        json.dump(data, fp, indent=4)

    elapsed = (time.time() - t0)
    diff = f"completed in {elapsed:.2f}s, {timedelta(seconds=elapsed)}"
    logging.info(diff)
    return diff, json.dumps(data, indent=4)

with gr.Blocks(title="Gradio", analytics_enabled=False) as demo:
    file_input = gr.File(label="select a audio file", file_types=[".wav"])

    output = gr.Textbox(label="execution time", interactive=False)
    output_json = gr.Textbox(label="output", interactive=False, lines=10)

    submit_button = gr.Button("Submit")
    submit_button.click(main, inputs=[file_input], outputs=[output, output_json])
    demo.launch(server_name="0.0.0.0", server_port=8080, inbrowser=True, debug=True, share=False)