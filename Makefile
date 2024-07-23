SRC = $(wildcard *.cpp Utils/*.cpp)
HEADERS = $(*.h Utils/*.h)
OBJ = ${SRC:.cpp=.o}

CXX = g++

main: ${OBJ}
	${CXX} ${CFLAGS}  $^ -o $@ -lm

%.o: %.c ${HEADERS}
	${CXX} ${CFLAGS} -c  $<  -o $@ -lm

clean:
	rm -f *.o
