import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=colors.HexColor("#1F3A5F"), fontSize=14, spaceBefore=10)
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=10, leading=14)
NOTE = ParagraphStyle("N", parent=B, backColor=colors.HexColor("#FFF4D6"), borderPadding=6, spaceBefore=6, spaceAfter=6)
S = []
def p(t, st=B): S.append(Paragraph(t, st)); S.append(Spacer(1, 5))
def h(t): S.append(Paragraph(t, H1))
def tbl(rows, w):
    t = Table(rows, colWidths=w, hAlign="LEFT")
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1F3A5F")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("GRID",(0,0),(-1,-1),0.4,colors.grey),("FONTSIZE",(0,0),(-1,-1),9),("VALIGN",(0,0),(-1,-1),"TOP")]))
    S.append(t); S.append(Spacer(1, 8))
def c(t): return Paragraph(t, ParagraphStyle("c", parent=B, fontSize=9, leading=12))

S.append(Paragraph("Cartoonify Image using GANs", ParagraphStyle("T", parent=ss["Title"], textColor=colors.HexColor("#1F3A5F"))))
p("<b>Project report</b> | Python, OpenCV, TensorFlow<br/>Author: Namani Umesh Chandra (co-author with Gangam Keerthika and Thadakamalla Abhishek; guide: Jayendra Kumar)<br/>"
  "Published as: <i>Cartoonify Image Using GAN</i>, IJISRT, Vol. 8, Issue 3, March 2023.")
p("<b>About this report:</b> it describes the project design and what this repository contains. Numbers for GAN training "
  "(dataset size, quality score, latency change) are left for the author to fill in from their own runs; none are invented here.", NOTE)

h("1. The idea in simple words")
p("Making cartoons by hand takes a lot of time and skill, and tools like Photoshop are paid and hard for beginners. "
  "This project turns an ordinary photo, GIF or video into a cartoon-style one automatically, and adds a few handy "
  "extras (denoising, upscaling, brightness, video to GIF) so it works as a small all-in-one image tool.")

h("2. How it works")
p("<b>Step A, prepare the image.</b> Noise is removed and the image can be enlarged using OpenCV.<br/>"
  "<b>Step B, cartoonize.</b> Two options are included: (1) a classic filter that finds edges, flattens colours and "
  "draws the edges back on top; (2) a GAN, a pair of neural networks where a <i>generator</i> makes cartoon versions "
  "and a <i>discriminator</i> tries to tell them apart from real cartoons, so the generator gradually learns the cartoon look.<br/>"
  "<b>Step C, video.</b> A video is split into frames, each frame is cartoonized, and the frames are joined again "
  "(or turned into a GIF).")
tbl([["Component", "Details"],
     [c("Generator"), c("Convolution stem, 2 down-sampling layers, 6 residual blocks, 2 up-sampling layers, tanh output")],
     [c("Discriminator"), c("PatchGAN: judges small patches as real cartoon or fake")],
     [c("Losses"), c("Least-squares adversarial loss + content loss on VGG19 features (keeps the photo's structure)")],
     [c("Augmentation"), c("Random crop, horizontal flip, small brightness change")],
     [c("Web app (paper)"), c("Django + MySQL front end with login, image and video pages (as described in the paper)")]],
    [4*cm, 12.5*cm])

h("3. What was run and checked")
tbl([["Item", "Status"],
     [c("Classic cartoon filter, denoise, upscale, brightness"), c("Implemented and run; 5 unit tests pass")],
     [c("Video to cartoon video, video to GIF, command line tool"), c("Implemented and run on a test clip")],
     [c("Latency benchmark script"), c("Run for the classic filter on a 512x512 image: about 80 ms per image on the build machine (CPU)")],
     [c("GAN model, training, inference code"), c("Written and syntax-checked; not trained in this repo (needs data and GPU)")],
     [c("GAN quality score and latency improvement"), c("To be filled in by the author from real training runs")]],
    [7*cm, 9.5*cm])

h("4. Example output (classic filter)")
S.append(Image("samples/side_by_side.png", width=15*cm, height=7.5*cm))
p("Left: original. Right: cartoonized. Test image: NASA astronaut photo from scikit-image (public domain).")

h("5. How to measure the results properly")
p("<b>Style quality.</b> Pick a fixed set of held-out photos, cartoonize them, and score them. Options: a simple "
  "rating survey (for example 20 people rate 'looks like a cartoon' and 'still looks like the original' from 1 to 5), "
  "or FID against a set of real cartoon frames. State exactly how the score was computed.<br/>"
  "<b>Latency.</b> Use <font face='Courier'>scripts/benchmark_latency.py</font> before and after an optimisation, on the same machine, "
  "and report the median time per image.")

h("6. Limitations and next steps")
p("The classic filter is quick but only a fixed effect. The GAN needs good data and tuning and can leave artefacts. "
  "Video output has no audio unless it is re-attached. Next steps: train on a larger cartoon set, add an edge-smoothing loss, "
  "export to TFLite for faster inference, and wrap everything in a simple web demo.")

h("7. References")
p("1. Y. Chen, Y.-K. Lai, Y.-J. Liu, CartoonGAN: Generative Adversarial Networks for Photo Cartoonization.<br/>"
  "2. I. Goodfellow et al., Generative Adversarial Nets, NIPS 2014.<br/>"
  "3. Namani U. C. et al., Cartoonify Image Using GAN, IJISRT Vol. 8 Issue 3, March 2023 (included in docs/).")
SimpleDocTemplate("reports_Cartoonify_Report.pdf", pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm,
                  title="Cartoonify Image using GANs - Report", author="Namani Umesh Chandra").build(S)
