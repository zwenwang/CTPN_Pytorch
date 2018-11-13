all:
	cython other/lib/nms.pyx
	gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.7 -o other/lib/nms.so other/lib/nms.c
	rm -rf other/lib/nms.c

	cython other/lib/find_top_bottom.pyx
	gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.7 -o other/lib/find_top_bottom.so other/lib/find_top_bottom.c
	rm -rf other/lib/find_top_bottom.c
