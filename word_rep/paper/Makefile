DOC_NAME    := wordpred

PS2PDF      := ps2pdf14

#use the following options to avoid the embedded font problem.
DVIPS       := dvips -j0 -Ppdf -Pdownload35 -G0 -t letter -D 1200 -Z -mode ljfzzz
LATEX       := latex
FIG2DEV     := fig2dev

#use the following options to generate into a single html.
LATEX2HTML  := latex2html -split 0 -split +0 -link 0 -no_navigation -no_footnode -show_section_numbers -local_icons
DOXYGEN     := doxygen
BIBTEX	    := bibtex

BASE := Makefile $(wildcard *.tex) $(wildcard *.bib)
BASE := $(BASE) $(wildcard ./pic/*.eps)

PSOPT 	:=  -dNOPAUSE -dBATCH \
	-sDEVICE=pdfwrite -dPDFSETTINGS=/prepress\
	-dCompatibilityLevel=1.5 \
	-dCompressPages=true -dUseFlateCompression=false \
	-sPAPERSIZE=letter \
	-dSubsetFonts=true -dEmbedAllFonts=true \
	-dProcessColorModel=/DeviceGray \
	-dDetectBlends=true -dOptimize=true \
	-dDownsampleColorImages=true -dColorImageResolution=1200 \
	-dColorImageDownsampleType=/Average -dColorImageFilter=/FlateEncode \
	-dAutoFilterColorImages=false -dAntiAliasColorImages=false \
	-dColorImageDownsampleThreshold=1.50000 \
	-dDownsampleGrayImages=true -dGrayImageResolution=1200 \
	-dGrayImageDownsampleType=/Average -dGrayImageFilter=/FlateEncode \
	-dAutoFilterGrayImages=false -dAntiAliasGrayImages=false \
	-dGrayImageDownsampleThreshold=1.50000 \
	-dDownsampleMonoImages=true -dMonoImageResolution=1200 \
	-dMonoImageDownsampleType=/Average -dMonoImageFilter=/FlateEncode \
	-dAutoFilterMonoImages=false -dAntiAliasMonoImages=false \
	-dMonoImageDownsampleThreshold=1.50000

all: pdf

pdf: $(DOC_NAME).pdf

%.pdf: %.ps
	$(PS2PDF) $(PSOPT) $< $@
	@cat $(DOC_NAME).log | grep Warning

%.ps: %.dvi
	$(DVIPS) $< -o $@

%.dvi: $(BASE)
	@make clean
	#@make -C eval all
	$(LATEX) $*.tex
	$(BIBTEX) $*
	$(LATEX) $*.tex
	$(BIBTEX) $*
	$(LATEX) $*.tex

paper:
	pdflatex wordpred.tex

html: $(DOC_NAME)/$(DOC_NAME).html

%.html: $(BASE)
	@rm -rf $(DOC_NAME)
	$(LATEX2HTML) $*.tex

clean:
	@rm -rf *.pdf *.ps *.dvi *.aux *.log *.bbl  *.blg *.idx *.out *~ pic/*~ $(DOC_NAME)
	#@make -C eval clean

.PHONY: clean all pdf html

### Makefile ends here
