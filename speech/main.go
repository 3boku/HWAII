package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"

	speech "cloud.google.com/go/speech/apiv1"
	"cloud.google.com/go/speech/apiv1/speechpb"
	"github.com/gin-gonic/gin"
)

func main() {
	r := gin.Default()

	// 정적 파일 (HTML, JavaScript) 제공
	r.Static("/", "./public") // 'public' 폴더에 웹사이트 파일들을 넣습니다.

	// 음성 변환 API 엔드포인트
	r.POST("/transcribe", transcribeHandler)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
		log.Printf("Defaulting to port %s", port)
	}
	r.Run(":" + port)
}

func transcribeHandler(c *gin.Context) {
	file, err := c.FormFile("audio")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "오디오 파일이 없습니다."})
		return
	}

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

	c.JSON(http.StatusOK, gin.H{"transcript": transcript})
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
	return transcript, nil
}
