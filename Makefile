all:
	# in manylinux2014 docker
	cd direct_mmap/cpp && g++ -pthread -DNDEBUG -w -O3 -fwrapv -shared -fPIC -I/opt/python/cp310-cp310/include/python3.10 -o memmap.so memmap.cpp

clean:
	rm -f direct_mmap/cpp/memmap.so

pipclean:
	rm -rf build/ dist/ *.egg-info/

pip:
	rm -rf build/ dist/ *.egg-info/
	python setup.py sdist build
	cp dist/*.tar.gz ./
	rm -rf build/ dist/ *.egg-info/
