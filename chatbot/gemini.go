package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"sync"

	"github.com/google/generative-ai-go/genai"
	"github.com/joho/godotenv"
	"google.golang.org/api/option"
)

// 전역 변수로 클라이언트와 모델 선언
var (
	client         *genai.Client
	model          *genai.GenerativeModel
	scriptData     []byte
	scriptMimeType string
	wikiData       []byte
	wikiMimeType   string
	once           sync.Once
	initErr        error
)

// 초기화 함수 추가
func initResources() error {
	// .env 파일 로드
	if err := godotenv.Load(); err != nil {
		return err
	}

	// API 키 가져오기
	apiKey := os.Getenv("GEMINI_API_KEY")
	ctx := context.Background()

	// 클라이언트 생성
	var err error
	client, err = genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		return err
	}

	// 모델 설정
	instructions := os.Getenv("GEMINI_INSTRUCTIONS")
	model = client.GenerativeModel("gemini-2.0-flash")
	model.SystemInstruction = &genai.Content{
		Parts: []genai.Part{genai.Text(instructions)},
	}
	model.SafetySettings = []*genai.SafetySetting{
		{
			Category:  genai.HarmCategoryHarassment,
			Threshold: genai.HarmBlockNone,
		},
		{
			Category:  genai.HarmCategoryHateSpeech,
			Threshold: genai.HarmBlockNone,
		},
		{
			Category:  genai.HarmCategoryDangerousContent,
			Threshold: genai.HarmBlockNone,
		},
		{
			Category:  genai.HarmCategorySexuallyExplicit,
			Threshold: genai.HarmBlockNone,
		},
	}

	// 스크립트 파일 로드
	scriptData, err = os.ReadFile("script.txt")
	if err != nil {
		return err
	}
	scriptMimeType = http.DetectContentType(scriptData)

	// 위키 파일 로드
	wikiData, err = os.ReadFile("nino_wiki.pdf")
	if err != nil {
		return err
	}
	wikiMimeType = http.DetectContentType(wikiData)

	return nil
}

func ChatWithNino(text string) genai.Part {
	// 한 번만 초기화
	once.Do(func() {
		initErr = initResources()
	})

	if initErr != nil {
		log.Fatal(initErr)
	}

	ctx := context.Background()
	cs := model.StartChat()

	// 사전에 로드된 데이터로 히스토리 설정
	cs.History = []*genai.Content{
		{
			Parts: []genai.Part{
				genai.Blob{
					MIMEType: scriptMimeType,
					Data:     scriptData,
				},
				genai.Text("니노의 말투입니다."),
				genai.Blob{
					MIMEType: wikiMimeType,
					Data:     wikiData,
				},
				genai.Text("니노의 정보입니다."),
			},
			Role: "model",
		},
	}

	resp, err := cs.SendMessage(ctx, genai.Text(text))
	if err != nil {
		log.Fatal(err)
	}

	return PrintModelResp(resp)
}

func PrintModelResp(resp *genai.GenerateContentResponse) genai.Part {
	var content genai.Part
	for _, cand := range resp.Candidates {
		if cand.Content != nil {
			for _, part := range cand.Content.Parts {
				content = part
			}
		}
	}
	return content
}

// 프로그램 종료 시 클라이언트 닫기 위한 함수
func CloseClient() {
	if client != nil {
		client.Close()
	}
}
