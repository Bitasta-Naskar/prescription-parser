# Dataset Observations — Doctor's Handwritten Prescription (BD) Dataset

Date: 29.8.26
Images reviewed: <!-- e.g. 25 of ~500 -->
Tool used: view_samples.py (contact sheet + a few full-size opens)
n/s:50/100
## Capture method
- [ ] Scanned (flat, even lighting, straight edges)
- [ ] Phone photo (perspective skew, shadows, uneven lighting)
- [ ] Mixed — note rough proportion: ___
50/100:
- scanned: flat, even lighting, mixed edges
50/200:flat, uneven lighting, mixed edges
50/300,50/400: phone photo, shadow, scanned flat, even lighting, mixed edges

## Background / template
- [ ] Blank paper
- template overlaps
50/100,50/200:
moderate to minimal overlap

## Lighting / image quality issues
- [ ] Shadows (from phone/hand)
- [ ] Glare / reflections
- [ ] Blur (motion or focus)
- [ ] Low contrast between ink and paper:low in 50/400
- [ ] Color paper or colored ink (not just black/blue on white)
- [ ] Skew / rotation — pages not aligned to frame
- [ ] Torn / creased / stained paper
-50/100,50/200
high contrast, skewed
50/300: reflection, high contrast, skewed

## Handwriting characteristics
- [ ] Cursive
- [ ] Block / print letters
- [ ] Mixed within same page
- Legibility (subjective, 1-5, 5=very clear): ___
- Are drug names abbreviated / shorthand-heavy?50/400
- Any overlapping/crossed-out text?50/200
- 50/100:Cursive, Block
50/200: crossed-out, cursive, block
50/300:Cursive, Block

## Layout
- Roughly how much of the image is actual prescription content vs margin/background?
- Is text organized in clear lines, or scattered across the page?
- Multiple languages/scripts present (e.g. English + Bengali)?

## Resolution / file characteristics
- Typical image dimensions: ___
- File formats present: ___
- Any images that are unusually large/small/corrupted?

## Implications for preprocessing (fill in after the above)
- Will need deskewing? Y/N
- Will need shadow/glare removal or adaptive thresholding? Y/N
- Will need to crop out printed template regions? Y/N
- Will need contrast enhancement / binarization? Y/N
- Any images to exclude from the initial pipeline (too damaged/illegible)?

## Open questions / things to check tomorrow
-
