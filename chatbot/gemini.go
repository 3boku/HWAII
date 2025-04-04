package main

import (
	"context"
	"log"
	"net/http"
	"os"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

type GeminiChatSession struct {
	ChatSession *genai.ChatSession
}

var ChatHistory []*genai.Content

func NewGeminiClient() *GeminiChatSession {
	apiKey := os.Getenv("GEMINI_API_KEY")
	ctx := context.Background()

	var err error
	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		log.Fatal(err)
	}

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

	ChatHistory = []*genai.Content{
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

	session := &GeminiChatSession{
		ChatSession: cs,
	}
	return session
}

func (cs *GeminiChatSession) ChatWithNino(ctx context.Context, text string) genai.Part {
	cs.ChatSession.History = ChatHistory

	resp, err := cs.ChatSession.SendMessage(ctx, genai.Text(text))
	if err != nil {
		log.Fatal(err)
	}

	var content genai.Part
	for _, cand := range resp.Candidates {
		if cand.Content != nil {
			for _, part := range cand.Content.Parts {
				content = part
			}
		}
	}

	ChatHistory = append(ChatHistory, &genai.Content{
		Parts: []genai.Part{
			genai.Text(text),
		},
		Role: "user",
	})
	ChatHistory = append(ChatHistory, &genai.Content{
		Parts: []genai.Part{
			content,
		},
		Role: "model",
	})

	return content
}
