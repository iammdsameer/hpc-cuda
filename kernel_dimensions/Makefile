CFLAGS = -Wno-deprecated-gpu-targets
CC = nvcc

all: 01 02 03 04 05 06 07 

01: 01.cu 
	$(CC) $(CFLAGS) -o 01 01.cu  

02: 02.cu 
	$(CC) $(CFLAGS) -o 02 02.cu 

03: 03.cu 
	$(CC) $(CFLAGS) -o 03 03.cu 

04: 04.cu 
	$(CC) $(CFLAGS) -o 04 04.cu 

05: 05.cu 
	$(CC) $(CFLAGS) -o 05 05.cu 

06: 06.cu 
	$(CC) $(CFLAGS) -o 06 06.cu 

07: 07.cu 
	$(CC) $(CFLAGS) -o 07 07.cu 
clean:
	rm -f 01 01 02 03 04 05 06 07 *~ a.out
