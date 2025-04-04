package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	speech "cloud.google.com/go/speech/apiv1"
	"cloud.google.com/go/speech/apiv1/speechpb"
	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/joho/godotenv"
)

func main() {
	r := gin.Default()

	if err := godotenv.Load(); err != nil {
		log.Printf("Warning: .env file not found, using environment variables")
	}

	r.Use(cors.New(cors.Config{
		AllowOrigins: []string{"*"},
		AllowMethods: []string{"GET", "POST", "PUT", "PATCH", "DELETE"},
		AllowHeaders: []string{"Origin", "Content-Type"},
		MaxAge:       24 * time.Hour,
	}))

	r.LoadHTMLGlob("templates/*.html")

	// 정적 파일 제공 경로 추가
	r.Static("/templates", "./templates")

	// 루트 경로에 index.html 제공
	r.GET("/", func(c *gin.Context) {
		c.HTML(http.StatusOK, "index.html", nil)
	})

	cs := NewGeminiClient()

	r.POST("/chat", func(c *gin.Context) {
		// 파일이 있는지 먼저 확인
		file, err := c.FormFile("audio")
		if err == nil {
			// 오디오 파일이 있는 경우 음성 처리
			log.Printf("수신된 파일 크기: %d 바이트, 타입: %s", file.Size, file.Header.Get("Content-Type"))

			src, err := file.Open()
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": "오디오 파일 열기 실패"})
				return
			}
			defer src.Close()

			audioBytes, err := io.ReadAll(src)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": "오디오 파일 읽기 실패"})
				return
			}

			transcript, err := transcribeAudio(c.Request.Context(), audioBytes)
			if err != nil {
				log.Printf("음성 변환 실패: %v", err)
				c.JSON(http.StatusInternalServerError, gin.H{"error": "음성 변환 실패"})
				return
			}

			log.Printf("음성 변환 결과: %s", transcript)
			resp := cs.ChatWithNino(c, transcript)
			c.JSON(http.StatusOK, gin.H{
				"message":    resp,
				"transcript": transcript,
			})
			return
		}

		// 오디오 파일이 없는 경우 JSON 메시지 확인
		var request struct {
			Message string `json:"message"`
		}

		if err := c.ShouldBindJSON(&request); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "잘못된 요청 형식"})
			return
		}

		if request.Message == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "메시지가 비어있습니다"})
			return
		}

		resp := cs.ChatWithNino(c, request.Message)
		c.JSON(http.StatusOK, gin.H{
			"message": resp,
		})
	})

	port := ":8080"
	log.Printf("서버 시작: http://localhost%s", port)
	r.Run(port)
}

func transcribeAudio(ctx context.Context, audioBytes []byte) (string, error) {
	client, err := speech.NewClient(ctx)
	if err != nil {
		return "", fmt.Errorf("speech.NewClient: %v", err)
	}
	defer client.Close()

	config := &speechpb.RecognitionConfig{
		Encoding:        speechpb.RecognitionConfig_WEBM_OPUS,
		SampleRateHertz: 48000,
		LanguageCode:    "ko-KR",
	}
	audio := &speechpb.RecognitionAudio{
		AudioSource: &speechpb.RecognitionAudio_Content{Content: audioBytes},
	}

	req := &speechpb.RecognizeRequest{
		Config: config,
		Audio:  audio,
	}

	resp, err := client.Recognize(ctx, req)
	if err != nil {
		return "", fmt.Errorf("Recognize: %v", err)
	}

	if len(resp.Results) == 0 {
		return "", fmt.Errorf("no recognition result returned")
	}

	var transcript string
	for _, result := range resp.Results {
		for _, alt := range result.Alternatives {
			transcript += alt.Transcript
		}
	}

	// 자주 잘못 인식되는 단어를 수정하는 후처리 로직 추가
	transcript = correctMisrecognizedWords(transcript)

	return transcript, nil
}

// 자주 잘못 인식되는 단어를 수정하는 함수
func correctMisrecognizedWords(text string) string {
	// 오인식 패턴 목록
	corrections := map[string]string{
		"민호야": "니노야",
		"민호":  "니노",
		"인호야": "니노야",
		"인호":  "니노",
		"이노야": "니노야",
		"이노":  "니노",
		"미노야": "니노야",
		"미노":  "니노",
		"리노야": "니노야",
		"리노":  "니노",
	}

	// 문자열 교체를 위한 정규식 활용 대신 직접 문자열 치환
	result := text
	for wrong, correct := range corrections {
		// 단순 문자열 치환(대소문자 구분)
		result = replaceWord(result, wrong, correct)
	}

	return result
}

// 전체 단어 단위로 치환하는 함수
func replaceWord(text, old, new string) string {
	// 문자열에서 old를 new로 치환
	// 단순 치환 대신 문맥을 고려한 치환을 위해 더 복잡한 로직 추가 가능

	// 여기서는 단순히 strings.Replace 사용
	return strings.Replace(text, old, new, -1)
}
