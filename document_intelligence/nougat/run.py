import re
import logging
from collections import defaultdict
from typing import List, Tuple, Dict
from PIL import Image
import fitz
import torch
from transformers import (
    VisionEncoderDecoderModel,
    AutoProcessor,
    StoppingCriteria,
    StoppingCriteriaList
)
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NougatEquation Task")

class RunningVarTorch:
    def __init__(self, L=15, norm=False):
        self.values = None
        self.L = L
        self.norm = norm

    def push(self, x: torch.Tensor):
        assert x.dim() == 1
        if self.values is None:
            self.values = x[:, None]
        elif self.values.shape[1] < self.L:
            self.values = torch.cat((self.values, x[:, None]), 1)
        else:
            self.values = torch.cat((self.values[:, 1:], x[:, None]), 1)

    def variance(self):
        if self.values is None:
            return
        if self.norm:
            return torch.var(self.values, 1) / self.values.shape[1]
        else:
            return torch.var(self.values, 1)


class StoppingCriteriaScores(StoppingCriteria):
    def __init__(self, threshold: float = 0.015, window_size: int = 200):
        super().__init__()
        self.threshold = threshold
        self.vars = RunningVarTorch(norm=True)
        self.varvars = RunningVarTorch(L=window_size)
        self.stop_inds = defaultdict(int)
        self.stopped = defaultdict(bool)
        self.size = 0
        self.window_size = window_size

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_scores = scores[-1]
        self.vars.push(last_scores.max(1)[0].float().cpu())
        self.varvars.push(self.vars.variance())
        self.size += 1
        if self.size < self.window_size:
            return False

        varvar = self.varvars.variance()
        for b in range(len(last_scores)):
            if varvar[b] < self.threshold:
                if self.stop_inds[b] > 0 and not self.stopped[b]:
                    self.stopped[b] = self.stop_inds[b] >= self.size
                else:
                    self.stop_inds[b] = int(
                        min(max(self.size, 1) * 1.15 + 150 + self.window_size, 4095)
                    )
            else:
                self.stop_inds[b] = 0
                self.stopped[b] = False
        return all(self.stopped.values()) and len(self.stopped) > 0


class ImageQueue:
    def __init__(self, images: List[Image.Image], batch_size: int):
        self.images = images
        self.batch_size = batch_size
        self.index = 0

    def next_batch(self) -> Tuple[List[Image.Image], int]:
        if self.index >= len(self.images):
            return None, self.index

        end_index = min(self.index + self.batch_size, len(self.images))
        batch = self.images[self.index:end_index]

        self.index += self.batch_size
        return batch, self.index

    def __len__(self):
      return len(self.images)

class NougatEquationModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}. Loading model on CPU until needed.")
        self.model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-small")
        self.processor = AutoProcessor.from_pretrained("facebook/nougat-small")

    def run_model(self, image_queue: ImageQueue):
        logger.info("Running model")
        logger.info(f"Using device: {self.device}")
        self.model.to(self.device)

        all_sequences = []

        while True:
            batch, index = image_queue.next_batch()
            if batch is None:
                break

            pixel_values = self.processor(images=batch, return_tensors="pt").pixel_values
            outputs = self.model.generate(
                pixel_values.to(self.device),
                min_length=1,
                max_new_tokens=3584,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                stopping_criteria=StoppingCriteriaList([StoppingCriteriaScores()]),
                return_dict_in_generate=True,
                output_scores=True,
            )
            sequences = self.processor.batch_decode(outputs['sequences'], skip_special_tokens=True)
            sequences = self.processor.post_process_generation(sequences, fix_markdown=False)

            all_sequences.extend(sequences)

            if index >= len(image_queue.images):
                break

        logger.info("Model run complete")
        logger.info("Returning model to CPU..")
        self.model.to("cpu")

        return all_sequences

    @staticmethod
    def extract_latex(text: str) -> List[str]:
        inline_pattern = re.compile(r'\\\(.*?\\\)')
        display_pattern = re.compile(r'\\\[.*?\\\]')
        dollar_pattern = re.compile(r'\$\$.*?\$\$', re.DOTALL)

        inline_latex = inline_pattern.findall(text)
        display_latex = display_pattern.findall(text)
        dollar_latex = dollar_pattern.findall(text)

        all_latex = inline_latex + display_latex + dollar_latex

        return list(set(all_latex))

def run_nougat(nougat_model: NougatEquationModel, images: List[Image.Image], batch_size: int) -> Dict[int, List[str]]:
    image_queue = ImageQueue(images, batch_size=batch_size)
    sequences = nougat_model.run_model(image_queue)
    latex_dict = {i: NougatEquationModel.extract_latex(sequence) for i, sequence in enumerate(sequences)}
    return latex_dict

app = FastAPI()

nougat_model = NougatEquationModel()

@app.post("/process_pdf")
async def process_pdf(file: UploadFile = File(...)):
    pdf_bytes = await file.read()

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for i in range(len(doc)):
        page = doc[i]
        image = page.get_pixmap()
        image = Image.frombytes("RGB", [image.width, image.height], image.samples)
        images.append(image)

    batch_size = 6
    latex_dict = run_nougat(nougat_model, images, batch_size)

    return JSONResponse(content=latex_dict)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
