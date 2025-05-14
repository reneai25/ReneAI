from typing import Annotated, List, Optional, Tuple

from surya.layout import LayoutPredictor
from surya.layout.schema import LayoutResult, LayoutBox

from marker.builders import BaseBuilder
from marker.providers.pdf import PdfProvider
from marker.schema import BlockTypes
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.schema.polygon import PolygonBox
from marker.schema.registry import get_block_class
from marker.settings import settings
from ultralytics import YOLO


class YOLOBuilder(BaseBuilder):
    """
    A builder for performing layout detection on PDF pages and merging the results into the document.
    """
    layout_batch_size: Annotated[
        Optional[int],
        "The batch size to use for the layout model.",
        "Default is None, which will use the default batch size for the model."
    ] = None
    force_layout_block: Annotated[
        str,
        "Skip layout and force every page to be treated as a specific block type.",
    ] = None
    disable_tqdm: Annotated[
        bool,
        "Disable tqdm progress bars.",
    ] = False

    def __init__(self, model_path: str, config=None):
        self.layout_model = YOLO(model_path)
        self.org={
            0:'Figure',
            2:'Table',
            3:'Text'
        }
        super().__init__(config)

    def __call__(self, document: Document, provider: PdfProvider):
        if self.force_layout_block is not None:
            # Assign the full content of every page to a single layout type
            layout_results = self.forced_layout(document.pages)
        else:
            layout_results = self.yolo_layout(document.pages)
        self.add_blocks_to_pages(document.pages, layout_results)

    def get_batch_size(self):
        if self.layout_batch_size is not None:
            return self.layout_batch_size
        elif settings.TORCH_DEVICE_MODEL == "cuda":
            return 6
        return 6

    def forced_layout(self, pages: List[PageGroup]) -> List[LayoutResult]:
        layout_results = []
        for page in pages:
            layout_results.append(
                LayoutResult(
                    image_bbox=page.polygon.bbox,
                    bboxes=[
                        LayoutBox(
                            label=self.force_layout_block,
                            position=0,
                            top_k={self.force_layout_block: 1},
                            polygon=page.polygon.polygon,
                        ),
                    ],
                    sliced=False
                )
            )
        return layout_results


    def yolo_layout(self, pages: List[PageGroup]) -> List[LayoutResult]:
        layout_results=[]
        for page in pages:
            image=page.get_image(highres=False)
            res=self.layout_model.predict(image)[0]
            layout_results.append((res.boxes.xyxy.tolist(),res.boxes.orig_shape,res.boxes.cls.tolist(),res.boxes.conf.tolist()))
        return layout_results

    def add_blocks_to_pages(self, pages: List[PageGroup], layout_results: List[Tuple[List[List[float]], Tuple[int, int], List[float], List[float]]]):
        for page, (bboxes,orig_shape,labels,confs) in zip(pages, layout_results):
            layout_page_height, layout_page_width = orig_shape
            layout_page_size = (layout_page_width, layout_page_height)
            provider_page_size = page.polygon.size
            page.layout_sliced = False # This indicates if the page was sliced by the layout model
            for i,bbox in enumerate(bboxes):
                x1,y1,x2,y2=bbox
                polygon_points=[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
                label_str = self.org.get(int(labels[i]), "Text")
                try:
                    label = BlockTypes[label_str]
                    confidence = float(confs[i])
                    layout_polygon = PolygonBox(polygon=polygon_points)
                    block_cls = get_block_class(label)
                    layout_block = page.add_block(block_cls, layout_polygon)
                    layout_block.polygon = layout_block.polygon.rescale(layout_page_size, provider_page_size)
                    layout_block.top_k = {label: confidence}
                    page.add_structure(layout_block)
                    
                except KeyError:
                    print(f"Warning: Block type '{label_str}' not found in BlockTypes registry.")
                except Exception as e:
                    print(f"Error adding block: {e}")
           
            # Ensure page has non-empty structure
            if page.structure is None:
                page.structure = []

            # Ensure page has non-empty children
            if page.children is None:
                page.children = []