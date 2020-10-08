CFLAGS = -Wno-deprecated-gpu-targets
CC = nvcc

all: dq00 dq01 dq02 dq03 

dq00: dq00.cu 
	$(CC) $(CFLAGS) -o dq00 dq00.cu  

dq01: dq01.cu 
	$(CC) $(CFLAGS) -o dq01 dq01.cu  

dq02: dq02.cu 
	$(CC) $(CFLAGS) -o dq02 dq02.cu 

dq03: dq03.cu 
	$(CC) $(CFLAGS) -o dq03 dq03.cu 

clean:
	rm -f dq00 dq01 dq01 dq02 dq03 *~ a.out
