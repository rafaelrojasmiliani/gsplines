
TEST=./tests/*.py
TEST=$(filter-out __init__.py,$(TEST))
all: $(SRC_TEX2:.tex=.pdf)
%.pdf: %.tex
	pdflatex $<
	convert -verbose -density 300  $@  -quality 100 -flatten -sharpen 0x1.0 -trim  +repage -transparent white ../$(@:.pdf=.jpg)
	-rm *.log *.aux
all:gsplines/*.py

