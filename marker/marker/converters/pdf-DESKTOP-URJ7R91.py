# Modified PdfConverter.__call__ method in pdf.py

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"# disables a tokenizers warning

from collections import defaultdict
from typing import Annotated, Any, Dict, List, Optional, Type, Tuple

from marker.processors import BaseProcessor
from marker.processors.llm.llm_table_merge import LLMTableMergeProcessor
from marker.providers.registry import provider_from_filepath
from marker.builders.document import DocumentBuilder
from marker.builders.yolo_layout import YOLOBuilder
from marker.builders.llm_yolo_layout import YOLO_LLMBuilder
from marker.builders.layout import LayoutBuilder
from marker.builders.llm_layout import LLMLayoutBuilder
from marker.builders.line import LineBuilder
from marker.builders.ocr import OcrBuilder
from marker.builders.structure import StructureBuilder
from marker.converters import BaseConverter
from marker.output import text_from_rendered
from marker.processors.blockquote import BlockquoteProcessor
from marker.processors.code import CodeProcessor
from marker.processors.debug import DebugProcessor
from marker.processors.document_toc import DocumentTOCProcessor
from marker.processors.equation import EquationProcessor
from marker.processors.footnote import FootnoteProcessor
from marker.processors.ignoretext import IgnoreTextProcessor
from marker.processors.line_numbers import LineNumbersProcessor
from marker.processors.list import ListProcessor
from marker.processors.llm.llm_complex import LLMComplexRegionProcessor
from marker.processors.llm.llm_form import LLMFormProcessor
from marker.processors.llm.llm_image_description import LLMImageDescriptionProcessor
from marker.processors.llm.llm_table import LLMTableProcessor
from marker.processors.llm.llm_inlinemath import LLMInlineMathLinesProcessor
from marker.processors.page_header import PageHeaderProcessor
from marker.processors.reference import ReferenceProcessor
from marker.processors.sectionheader import SectionHeaderProcessor
from marker.processors.table import TableProcessor
from marker.processors.text import TextProcessor
from marker.processors.llm.llm_equation import LLMEquationProcessor
from marker.renderers.markdown import MarkdownRenderer
from marker.schema import BlockTypes
from marker.schema.blocks import Block
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.schema.registry import register_block_class
from marker.util import strings_to_classes
from marker.processors.llm.llm_handwriting import LLMHandwritingProcessor
from marker.processors.order import OrderProcessor
from marker.services.gemini import GoogleGeminiService
from marker.processors.line_merge import LineMergeProcessor
from marker.processors.llm.llm_mathblock import LLMMathBlockProcessor
from shapely.geometry import Polygon
import re
import os
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher
from PIL import Image as PILImage

class PdfConverter(BaseConverter):
    """
    A converter for processing and rendering PDF files into Markdown, JSON, HTML and other formats.
    """
    override_map: Annotated[
        Dict[BlockTypes, Type[Block]],
        "A mapping to override the default block classes for specific block types.",
        "The keys are `BlockTypes` enum values, representing the types of blocks,",
        "and the values are corresponding `Block` class implementations to use",
        "instead of the defaults."
    ] = defaultdict()
    use_llm: Annotated[
        bool,
        "Enable higher quality processing with LLMs.",
    ] = False
    default_processors: Tuple[BaseProcessor, ...] = (
        OrderProcessor,
        LineMergeProcessor,
        BlockquoteProcessor,
        CodeProcessor,
        DocumentTOCProcessor,
        EquationProcessor,
        FootnoteProcessor,
        IgnoreTextProcessor,
        LineNumbersProcessor,
        ListProcessor,
        PageHeaderProcessor,
        SectionHeaderProcessor,
        TableProcessor,
        LLMTableProcessor,
        LLMTableMergeProcessor,
        LLMFormProcessor,
        TextProcessor,
        LLMInlineMathLinesProcessor,
        LLMComplexRegionProcessor,
        LLMImageDescriptionProcessor,
        LLMEquationProcessor,
        LLMHandwritingProcessor,
        LLMMathBlockProcessor,
        ReferenceProcessor,
        DebugProcessor,
    )

    def __init__(
        self,
        artifact_dict: Dict[str, Any],
        processor_list: Optional[List[str]] = None,
        renderer: str | None = None,
        llm_service: str | None = None,
        config=None
    ):
        super().__init__(config)

        if config is None:
            config = {}

        for block_type, override_block_type in self.override_map.items():
            register_block_class(block_type, override_block_type)

        if processor_list:
            processor_list = strings_to_classes(processor_list)
        else:
            processor_list = self.default_processors

        if renderer:
            renderer = strings_to_classes([renderer])[0]
        else:
            renderer = MarkdownRenderer

        if llm_service:
            llm_service_cls = strings_to_classes([llm_service])[0]
            llm_service = self.resolve_dependencies(llm_service_cls)
        elif config.get("use_llm", False):
            llm_service = self.resolve_dependencies(GoogleGeminiService)

        # Inject llm service into artifact_dict so it can be picked up by processors, etc.
        artifact_dict["llm_service"] = llm_service
        self.llm_service = llm_service

        self.artifact_dict = artifact_dict
        self.renderer = renderer

        processor_list = self.initialize_processors(processor_list)
        self.processor_list = processor_list
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2') # Load semantic model

    def build_document(self, filepath: str, layout_builder):
        provider_cls = provider_from_filepath(filepath)
        line_builder = self.resolve_dependencies(LineBuilder)
        ocr_builder = self.resolve_dependencies(OcrBuilder)
        provider = provider_cls(filepath, self.config)
        document = DocumentBuilder(self.config)(provider, layout_builder, line_builder, ocr_builder)
        structure_builder_cls = self.resolve_dependencies(StructureBuilder)
        structure_builder_cls(document)

        for processor in self.processor_list:
            processor(document)

        return document


    def calculate_iou(self,bbox1, bbox2):
        """Calculates the Intersection over Union (IoU) of two bounding boxes.

        Args:
            bbox1: A tuple or list representing the first bounding box (x_min, y_min, x_max, y_max).
            bbox2: A tuple or list representing the second bounding box (x_min, y_min, x_max, y_max).

        Returns:
            The IoU value (float) between 0 and 1.
        """
        poly1 = Polygon([(bbox1[0], bbox1[1]), (bbox1[2], bbox1[1]), (bbox1[2], bbox1[3]), (bbox1[0], bbox1[3])])
        poly2 = Polygon([(bbox2[0], bbox2[1]), (bbox2[2], bbox2[1]), (bbox2[2], bbox2[3]), (bbox2[0], bbox2[3])])

        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        iou = intersection / union if union > 0 else 0
        return iou


    def __call__(self, filepath: str):
        renderer = self.resolve_dependencies(self.renderer)
        self.layout_builder_class = LayoutBuilder
        if self.use_llm:
            self.layout_builder_class = LLMLayoutBuilder
        layout_builder_llm = self.resolve_dependencies(self.layout_builder_class)
        document_llm = self.build_document(filepath, layout_builder_llm)
        text_llm = renderer(document_llm).markdown
        layout_builder_yolo = YOLOBuilder(model_path=self.config.get("yolo_model_path"), config=self.config)
        if self.use_llm:
            layout_builder_yolo = YOLO_LLMBuilder(model_path=self.config.get("yolo_model_path"), llm_service=self.llm_service, config=self.config)
        document_yolo = self.build_document(filepath, layout_builder_yolo)
        yolo_llm= renderer(document_yolo)
        yolo=yolo_llm.markdown
        marker_blocks=[block for page in document_llm.pages for block_id in page.structure for block in [page.get_block(block_id)]]
        yolo_blocks=[block for page in document_yolo.pages for block_id in page.structure for block in [page.get_block(block_id)]]
        matched_elements = []
        for marker_block in marker_blocks:
            best_match_yolo = None
            highest_similarity = -1
            highest_iou = -1

            for yolo_block in yolo_blocks:
                # Simple matching based on text similarity (you'll need to refine this)
                similarity = SequenceMatcher(None, getattr(marker_block, 'text', ''), getattr(yolo_block, 'text', '')).ratio()
                if similarity > highest_similarity and similarity > 0.7: # Example threshold
                    highest_similarity = similarity
                    best_match_yolo = yolo_block

            if best_match_yolo:
                marker_bbox = marker_block.polygon.bbox
                yolo_bbox = best_match_yolo.polygon.bbox
                iou = self.calculate_iou(marker_bbox, yolo_bbox)

                matched_elements.append({
                    'marker': marker_block,
                    'yolo': best_match_yolo,
                    'similarity': highest_similarity,
                    'iou': iou
                })
        iou_threshold = 0.6 # You can adjust this threshold

        # Create a mapping of marker_block to its best matching yolo_block and IoU
        marker_to_yolo_iou = {match['marker'].block_id: (match['yolo'], match['iou']) for match in matched_elements}
        merged_blocks = []
        matched_yolo_blocks = set() # Keep track of matched YOLO blocks

        marker_block_map = {block.block_id: block for block in marker_blocks} # Create a map for easy access to marker blocks

        for marker_block_id in marker_block_map:
            marker_block = marker_block_map[marker_block_id]
            if marker_block_id in marker_to_yolo_iou:
                yolo_block, iou = marker_to_yolo_iou[marker_block_id]
                if iou > iou_threshold:
                    # For high IoU, you might choose to use YOLO's bounding box
                    marker_block.polygon.polygon = yolo_block.polygon.polygon
                    marker_block.polygon.bbox = yolo_block.polygon.bbox
                merged_blocks.append(marker_block)
                matched_yolo_blocks.add(yolo_block.block_id)
            else:
                # If no match or IoU is low, keep the original marker block
                merged_blocks.append(marker_block.block_id)

        # Add unmatched YOLO blocks (optional - you might want to be careful with this)
        # for yolo_block in yolo_blocks:
        #     if yolo_block not in matched_yolo_blocks:
        #         # You might want to further process these, e.g., check if they are unique elements
        #         merged_blocks.append(yolo_block)

        # For now, let's just return the merged blocks
        return merged_blocks