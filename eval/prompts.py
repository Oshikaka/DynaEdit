"""论文附录 F 的 VLM 评测 prompt，逐字照搬（仅去掉 PDF 抽取带来的换行/缩进痕迹）。

三个维度各打 1-5 分：
  CONTENT_PRESERVATION —— 对源视频的保真（需要 源视频 + 编辑视频 + 两条 prompt）
  TEXT_ADHERENCE       —— 对目标 prompt 的符合（需要 编辑视频 + target prompt）
  VISUAL_QUALITY       —— 视觉质量（只需要 编辑视频）
"""

CONTENT_PRESERVATION_INSTRUCTIONS = """
### Task Definition
You will be provided with:
SOURCE_VIDEO
SOURCE_PROMPT - a text description of SOURCE_VIDEO
TARGET_VIDEO
TARGET_PROMPT - a text description of TARGET_VIDEO

You need to carefully analyze the difference between SOURCE_PROMPT and TARGET_PROMPT to understand what should be the difference between SOURCE_VIDEO and TARGET_VIDEO.
TARGET_VIDEO should be a minimal edit of SOURCE_VIDEO to make it adhere to TARGET_PROMPT. Elements in SOURCE_VIDEO that are not required to change to match TARGET_PROMPT should remain the same in TARGET_VIDEO.
As an experienced evaluator, your task is to evaluate how well the edited video TARGET_VIDEO is consistent with SOURCE_VIDEO, according to the scoring criteria.

### Scoring Criteria
Unless a modification is necessary to align with TARGET_PROMPT, the following criteria should be met:
1. Objects: Determine if the elements and subjects in SOURCE_VIDEO are presented in TARGET_VIDEO.
2. Alignment - the content of TARGET_VIDEO should align with the content of SOURCE_VIDEO both spatially and temporally.
3. Motions and Actions: The motions and actions in TARGET_VIDEO should be as similar to the ones in SOURCE_VIDEO as possible.
4. Camera Motion: The camera trajectory in should be identical to the one in SOURCE_VIDEO.
Do not take into consideration visual quality. Ignore visual artifacts.

### Scoring Range
Based on these criteria, a specific integer score from 1 to 5 can be assigned to determine the level of semantic consistency:
Very Poor (1): No consistency. TARGET_VIDEO completely and unneededly deviates from SOURCE_VIDEO, one could not understand from watching TARGET_VIDEO that its source was SOURCE_VIDEO.
Poor (2): Weak consistency. TARGET_VIDEO resembles SOURCE_VIDEO semantically (same types of objects, motion, etc) but the visual content is not consistent.
Fair (3): Moderate consistency. TARGET_VIDEO resembles SOURCE_VIDEO to an extent but lacks several important details or contains some inaccuracies.
Good (4): Strong consistency. TARGET_VIDEO accurately depicts most of the information from the SOURCE_VIDEO with only minor omissions or inaccuracies.
Excellent (5): Near-perfect consistency. TARGET_VIDEO remains similar to SOURCE_VIDEO with high precision and detail, with no modification other that what is necessary to adhere to TARGET_PROMPT.

### Output format
Provide only a score in the range of 1-5, and nothing else.

## Inputs
SOURCE_PROMPT: {source_prompt}
TARGET_PROMPT: {target_prompt}
"""

TEXT_ADHERENCE_INSTRUCTIONS = """
### Task Definition
You will be provided with a VIDEO, and text PROMPT describing its content.
As an experienced evaluator, your task is to evaluate the semantic consistency between VIDEO and PROMPT, according to the scoring criteria.

### Scoring Criteria
When assessing the semantic consistency between an image and its accompanying PROMPT, it is crucial to consider how well the visual content of the VIDEO aligns with the textual description in PROMPT. This evaluation can be based on several key aspects:
1. Relevance: Determine if the elements and subjects presented in the VIDEO directly relate to the core topics and concepts mentioned in the PROMPT. The VIDEO should reflect the main ideas or narratives described.
2. Accuracy: Examine the VIDEO for the presence and correctness of specific details mentioned in the PROMPT. This includes the depiction of particular objects, settings, actions, or characteristics that the PROMPT describes.
3. Completeness: Evaluate whether the VIDEO captures all the critical elements of the PROMPT. The VIDEO should not omit significant details that are necessary for the full understanding of the PROMPT's message.
4. Context: Consider the context in which the PROMPT places the subject and whether the VIDEO accurately represents this setting. This includes the portrayal of the appropriate environment, interactions, and background elements that align with the PROMPT.

### Scoring Range
Based on these criteria, a specific integer score from 1 to 5 can be assigned to determine the level of semantic consistency:
Very Poor (1): No correlation. The VIDEO does not reflect any of the key points or details of the PROMPT.
Poor (2): Weak correlation. The VIDEO addresses the PROMPT in a very general sense but misses most details and nuances.
Fair (3): Moderate correlation. The VIDEO represents the PROMPT to an extent but lacks several important details or contains some inaccuracies.
Good (4): Strong correlation. The VIDEO accurately depicts most of the information from the PROMPT with only minor omissions or inaccuracies.
Excellent (5): Near-perfect correlation. The VIDEO captures the PROMPT's content with high precision and detail, leaving out no significant information.

### Output format
Provide only a score in the range of 1-5, and nothing else.

## Inputs
PROMPT: {target_prompt}
"""

VISUAL_QUALITY_INSTRUCTIONS = """
### Task Definition
You will be provided with a VIDEO.
As an experienced evaluator, your task is to evaluate the visual quality of the video, according to the scoring criteria.

### Scoring Criteria
1. Bluriness: Is the video sharp or blurry?
2. artifacts: Are there artifacts in the video?
3. Flickerness: Is the video temporally smooth or exhibit temporal flickering?
4. temporal consistency: Is the video temporally consistent or do objects unreasonably change in time? unnatural motions are ok but not changes to the objects.

### Scoring Range
Based on these criteria, a specific integer score from 1 to 5 can be assigned to determine the level of visual quality:
Very Poor (1): the video suffers from severe artifacts, blurriness, flickeriness, or temporal inconsistencies.
Poor (2): the video suffers from many artifacts, blurriness, flickeriness, or temporal inconsistencies.
Fair (3): the video has noticable artifacts, blurriness, flickeriness, or temporal inconsistencies but overall quality is ok.
Good (4): the video has minor artifacts, blurriness, flickeriness, or temporal inconsistencies but overall quality is good.
Excellent (5): the video has no artifacts, blurriness, flickeriness, or temporal inconsistencies. It can pass as a legit video.

### Output format
Provide only a score in the range of 1-5, and nothing else.
"""

DIMENSIONS = {
    "content_preservation": CONTENT_PRESERVATION_INSTRUCTIONS,
    "text_adherence": TEXT_ADHERENCE_INSTRUCTIONS,
    "visual_quality": VISUAL_QUALITY_INSTRUCTIONS,
}
