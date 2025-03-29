package main

import (
	"bufio"
	"fmt"
	"os"
	"regexp"
	"strings"
)

func main() {
	var filteredScript string
	r := regexp.MustCompile(`\(二乃\)(.*)`)

	for i := 1; i <= 12; i++ {
		filename := fmt.Sprintf("files/[Judas] Go-Toubun no Hanayome - S02E%02d.srt", i)
		file, err := os.Open(filename)
		if err != nil {
			fmt.Println("파일 열기 오류:", err)
			continue
		}
		defer file.Close()

		scanner := bufio.NewScanner(file)

		for scanner.Scan() {
			line := scanner.Text()
			if matches := r.FindStringSubmatch(line); len(matches) > 1 {
				dialogue := strings.TrimSpace(matches[1])
				if dialogue != "" {
					// 에피소드 번호 없이 대사만 저장
					filteredScript += dialogue + "\n"
				}
			}
		}
	}

	err := os.WriteFile("script.txt", []byte(filteredScript), 0644)
	if err != nil {
		fmt.Println("파일 쓰기 오류:", err)
	} else {
		fmt.Println("script.txt 파일이 생성되었습니다.")
	}
}
