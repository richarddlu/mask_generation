LIBS = \
	-lopencv_core \
	-lopencv_highgui \
	-lopencv_imgproc

all:	demo

demo:	demo.o
		g++ demo.o -o demo $(LIBS)

demo.o:	demo.cpp
		g++ -c demo.cpp

clean:
	rm -f demo demo.o