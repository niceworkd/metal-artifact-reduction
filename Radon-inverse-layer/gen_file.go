package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
)

func main() {
	fileName := "test.txt"
	file, err := os.OpenFile(fileName, os.O_RDWR|os.O_CREATE, 0755)
	if err != nil {
		log.Fatal(err)
	}
	input := bufio.NewWriter(file)
	for i := 1; i <= 4000; i++ {
		input.WriteString(fmt.Sprintf("r%04d.mat s%04d.mat\n", i, i))
	}
	input.Flush()

	//input.WriteString()
}
