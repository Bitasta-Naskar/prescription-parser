# Dataset Observations — Doctor's Handwritten Prescription (BD) Dataset

Date: 29.8.26
Images reviewed: <n/s was from (50/100) to (50/1500)-->
Tool used: view_samples.py (contact sheet + a few full-size opens)
## Capture method
- Scanned (flat, even lighting, straight edges)
- Phone photo (perspective skew, shadows, uneven lighting)
- Mixed — note rough proportion: ___
50/100:
- scanned: flat, even lighting, mixed edges
50/200:flat, uneven lighting, mixed edges
50/300,50/400: phone photo, shadow, scanned flat, even lighting, mixed edges
50/500:flat, even lighting, mixed edges

## Background / template
- Blank paper
- template overlaps
50/100,50/200,50/500:moderate to minimal overlap
50/600: letters overlapping

## Lighting / image quality issues
- Shadows (from phone/hand)
- Glare / reflections
- Blur (motion or focus)
- Low contrast between ink and paper:low in 50/400
- Color paper or colored ink (not just black/blue on white)
- Skew / rotation — pages not aligned to frame
- Torn / creased / stained paper
-50/100,50/200:high contrast, skewed
50/300: reflection, high contrast, skewed
50/500: high contrast, skewed
low-high contrast(all)

## Handwriting characteristics
- Cursive
- Block / print letters
- Mixed within same page
- Legibility (subjective, 1-5, 5=very clear):1-5
- Are drug names abbreviated / shorthand-heavy?50/400, rest(not heavily done but yeah)
- Any overlapping/crossed-out text?50/200
- 50/100:Cursive, Block
50/200: crossed-out, cursive, block
50/300:Cursive, Block
50/500: crossed-out, cursive, block

## Layout
- Roughly how much of the image is actual prescription content vs margin/background?
- Is text organized in clear lines, or scattered across the page?
- Multiple languages/scripts present (e.g. English + Bengali)? jus english

## Resolution / file characteristics
- Typical image dimensions: ___
- File formats present: ___
- Any images that are unusually large/small/corrupted? some

## Implications for preprocessing (fill in after the above)
- Will need deskewing? Y/N->Y
- Will need shadow/glare removal or adaptive thresholding? Y/N->N
- Will need to crop out printed template regions? Y/N
- Will need contrast enhancement / binarization? Y/N
- Any images to exclude from the initial pipeline (too damaged/illegible)?

