package client

import (
	"bytes"
	"context"
	"fmt"
	"mime"

	"github.com/google/uuid"
	"github.com/sashabaranov/go-openai"
)

type AVFile struct {
	ContentType string // the audio or video mime type
	Content     []byte
}

// transcribes the audio of audio or video files
func TranscribeAV(c OpenAI, r TranscriptionRequest) (string, error) {
	validWhisperExtensions := map[string]bool{
		".m4a":  true,
		".mp3":  true,
		".webm": true,
		".mp4":  true,
		".mpga": true,
		".wav":  true,
		".mpeg": true,
	}
	exts, err := mime.ExtensionsByType(r.File.ContentType)
	if err != nil {
		return "", err
	}
	var fileExtension string
	for _, e := range exts {
		if validWhisperExtensions[e] {
			fileExtension = e
			break
		}
	}
	if len(fileExtension) == 0 {
		return "", fmt.Errorf("no file extension found for content type %q", r.File.ContentType)
	}
	t, err := c.CreateTranscription(context.Background(), openai.AudioRequest{
		Model:       openai.Whisper1,
		FilePath:    uuid.NewString() + fileExtension, // just needed for the extension
		Prompt:      r.Prompt,
		Reader:      bytes.NewReader(r.File.Content),
		Temperature: 1,
	})
	if err != nil {
		return "", err
	}
	return t.Text, nil
}

func init() {
	mime.AddExtensionType(".m4a", "audio/mp4")
	mime.AddExtensionType(".mp3", "audio/mp3")
	mime.AddExtensionType(".webm", "audio/webm")
	mime.AddExtensionType(".mp4", "video/mp4")
	mime.AddExtensionType(".mpga", "audio/mpeg")
	mime.AddExtensionType(".wav", "audio/wav")
	mime.AddExtensionType(".mpeg", "video/mpeg")
}
