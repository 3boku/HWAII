package main

import (
	"context"
	"github.com/joho/godotenv"
	"google.golang.org/api/option"
	"log"
	"net/http"
	"os"

	"github.com/google/generative-ai-go/genai"
)

func ChatWithNino(text string) genai.Part {
	// .env 파일 로드
	if err := godotenv.Load(); err != nil {
		log.Fatalf("Error loading .env file")
	}

	// API 키 가져오기
	apiKey := os.Getenv("GEMINI_API_KEY")
	ctx := context.Background()

	// 클라이언트 생성
	var err error
	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		log.Fatal(err)
	}

	// 모델 설정
	instructions := os.Getenv("GEMINI_INSTRUCTIONS")
	model := client.GenerativeModel("gemini-2.0-flash")
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
	scriptData, err := os.ReadFile("script.txt")
	if err != nil {
		log.Fatal(err)
	}
	scriptMimeType := http.DetectContentType(scriptData)

	// 위키 파일 로드
	wikiData, err := os.ReadFile("nino_wiki.pdf")
	if err != nil {
		log.Fatal(err)
	}
	wikiMimeType := http.DetectContentType(wikiData)

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
