LIBS = \
	-lopencv_core \
	-lopencv_highgui \
	-lopencv_imgproc

all:	demo

demo:	demo.o color_constraint.o nnls.o
		g++ demo.o color_constraint.o nnls.o -o demo $(LIBS)

demo.o:	demo.cpp
		g++ -c demo.cpp

color_constraint.o:	color_constraint.cpp
					g++ -c color_constraint.cpp

nnls.o:	nnls.cpp
		g++ -c nnls.cpp

clean:
	rm -f demo demo.o color_constraint.o nnls.o