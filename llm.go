// package llmif is an interface to llm's via json structs
package llm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"sort"
	"strings"

	"github.com/invopop/jsonschema"
	"github.com/sashabaranov/go-openai"
	"github.com/vincent-petithory/dataurl"
	"xoba.com/llm/assets"
	"xoba.com/llm/client"
	"xoba.com/llm/schema"
)

// Question is a question to ask the assistant
// ANSWER is the type of the answer, field names should be self-explanatory
type Question[ANSWER any] struct {
	Prompt   string          // the question to ask, including your prompt
	Files    []File          // files to use as background material
	Tools    map[string]Tool // tools at the assistant's disposal
	Examples []ANSWER        // examples of what the answer may look like
}

// File is a file to use as background material
type File struct {
	Name        string
	Content     []byte
	ContentType string
}

type Tool interface {
	Defintion() openai.FunctionDefinition
	Compute(parameters string) (string, error)
}

// Response is the response to asking a Question
type Response[ANSWER any] struct {
	Meta   string // free-form meta information about the process
	Answer ANSWER
}

func (r Response[T]) String() string {
	buf, _ := json.MarshalIndent(r, "", "  ")
	return string(buf)
}

func Ask[ANSWER any](c client.Interface, q Question[ANSWER]) (Response[ANSWER], error) {
	var zero Response[ANSWER]
	var messages []openai.ChatCompletionMessage
	messages = append(messages, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleSystem,
		Content: assets.Prompt,
	})
	whichModel := client.GPT4Turbo
	responseFormat := client.JSONResponse
	var maxTokens int
	if len(q.Files) > 0 {
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleSystem,
			Content: fmt.Sprintf(`there are going to be %d files in the following request, each of which you will use as background material for assisting the user.`, len(q.Files)),
		})
	}
	for _, d := range q.Files {
		switch d.ContentType {
		case "audio/mp3", "audio/mp4", "audio/mpeg", "audio/wav", "audio/x-wav", "audio/webm", "video/mp4", "video/mpeg", "video/webm":
			txt, err := c.TranscribeAV(client.TranscriptionRequest{
				File: client.AVFile{ContentType: d.ContentType, Content: d.Content},
			})
			if err != nil {
				return zero, err
			}
			messages = append(messages, openai.ChatCompletionMessage{
				Role: openai.ChatMessageRoleSystem,
				Content: fmt.Sprintf(
					"here is the transcription of a %s file named %q:\n\n%s",
					d.ContentType, d.Name,
					txt,
				),
			})
		case "application/pdf":
			txt, err := pdfToText(d.Content)
			if err != nil {
				return zero, err
			}
			messages = append(messages, openai.ChatCompletionMessage{
				Role: openai.ChatMessageRoleSystem,
				Content: fmt.Sprintf(
					"here is the text rendering of an %s file named %q:\n\n%s",
					d.ContentType, d.Name,
					string(txt),
				),
			})
		case "application/json",
			"text/plain", "text/html", "text/markdown", "text/csv", "text/xml", "text/rtf",
			"text/tab-separated-values", "text/richtext",
			"text/yaml", "text/x-yaml", "text/x-markdown", "text/x-rst", "text/x-org":
			messages = append(messages, openai.ChatCompletionMessage{
				Role: openai.ChatMessageRoleSystem,
				Content: fmt.Sprintf(
					"here is a %s file named %q:\n\n%s",
					d.ContentType, d.Name,
					string(d.Content),
				),
			})
		case "image/png", "image/jpeg", "image/webp", "image/gif":
			whichModel = client.GPT4Vision
			responseFormat = client.NoneSpecified // vision has no format at all
			maxTokens = 4096                      // specify, since vision defaults to few output tokens
			messages = append(messages, openai.ChatCompletionMessage{
				Role: openai.ChatMessageRoleSystem,
				MultiContent: []openai.ChatMessagePart{
					{
						Type: openai.ChatMessagePartTypeText,
						Text: fmt.Sprintf(
							"here is an %s file named %q",
							d.ContentType, d.Name,
						),
					},
					{
						Type: openai.ChatMessagePartTypeImageURL,
						ImageURL: &openai.ChatMessageImageURL{
							URL: dataurl.New(d.Content, d.ContentType).String(),
						},
					},
				},
			})
		default:
			return zero, fmt.Errorf("unsupported content type: %q", d.ContentType)
		}
	}
	messages = append(messages, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: q.Prompt,
	})
	{
		responseSchema := schema.Calculate(&Response[ANSWER]{})
		schema, err := json.MarshalIndent(responseSchema, "", "  ")
		if err != nil {
			return zero, err
		}
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: fmt.Sprintf(`the schema of your json answer must match: %s`, string(schema)),
		})
	}
	if len(q.Examples) > 0 {
		examples := new(bytes.Buffer)
		fmt.Fprintf(examples, "here are %d fictitious example(s) for how your json response may look like in practice:\n\n", len(q.Examples))
		for i, e := range q.Examples {
			a := Response[ANSWER]{
				Answer: e,
				Meta: cleanText(`this is a free-form response field for meta-level information, 
			it can be anything related to the task at hand or 
			the conversational process, not necessarily the question or answer per se.`),
			}
			buf, err := json.MarshalIndent(a, "", "  ")
			if err != nil {
				return zero, err
			}
			fmt.Fprintf(examples, "example #%d: %s\n\n", i+1, string(buf))
		}
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: examples.String(),
		})
	}
	var errs []error
	var tools []openai.Tool
	switch whichModel {
	case client.GPT4Turbo:
		for name, t := range q.Tools {
			def := t.Defintion()
			if name != def.Name {
				return zero, fmt.Errorf("tool name %q does not match definition name %q", name, def.Name)
			}
			tools = append(tools, openai.Tool{
				Type:     openai.ToolTypeFunction,
				Function: def,
			})
		}
		sort.Slice(tools, func(i, j int) bool {
			return tools[i].Function.Name < tools[j].Function.Name
		})
	case client.GPT4Vision:
		// tools not supported by GPTV!
	}
LOOP:
	for {
		if len(errs) > 4 {
			return zero, fmt.Errorf("too many tries: %v", errs)
		}
		resp, err := c.Complete(client.CompletionRequest{
			Model:     whichModel,
			Format:    responseFormat,
			MaxTokens: maxTokens,
			Stream:    os.Stdout,
			Messages:  messages,
			Tools:     tools,
		})
		if err != nil {
			return zero, err
		}
		switch resp.FinishReason {
		case "tool_calls":
			for _, call := range resp.FunctionCalls {
				tool, ok := q.Tools[call.Name]
				if !ok {
					return zero, fmt.Errorf("unknown tool: %q", call.Name)
				}
				result, err := tool.Compute(call.Arguments)
				if err != nil {
					return zero, err
				}
				fmt.Printf("result = %s\n", result)
				messages = append(messages, openai.ChatCompletionMessage{
					Role: openai.ChatMessageRoleAssistant,
					ToolCalls: []openai.ToolCall{
						{
							ID:   call.ID,
							Type: openai.ToolTypeFunction,
							Function: openai.FunctionCall{
								Name:      call.Name,
								Arguments: call.Arguments,
							},
						},
					},
				})
				messages = append(messages, openai.ChatCompletionMessage{
					Role:       openai.ChatMessageRoleTool,
					Content:    result,
					ToolCallID: call.ID,
				})
			}
			continue LOOP

		case "stop":
			messages = append(messages, openai.ChatCompletionMessage{
				Role:    openai.ChatMessageRoleAssistant,
				Content: resp.Content,
			})
			// minor cleanup if needed:
			resp.Content = strings.TrimSpace(resp.Content)
			resp.Content = strings.TrimPrefix(resp.Content, "```json")
			resp.Content = strings.TrimSuffix(resp.Content, "```")
			d := json.NewDecoder(strings.NewReader(resp.Content))
			d.DisallowUnknownFields()
			var parsedResponse Response[ANSWER]
			if err := d.Decode(&parsedResponse); err != nil {
				log.Printf("error decoding, going to potentially retry: %v", err)
				errs = append(errs, err)
				messages = append(messages, openai.ChatCompletionMessage{
					Role:    openai.ChatMessageRoleUser,
					Content: fmt.Sprintf("oops, i got an error parsing your response as json: %v. could you please re-do with correct json having no extraneous characters, etc?", err),
				})
				continue LOOP
			}
			return parsedResponse, nil
		default:
			return zero, fmt.Errorf("unhandled finish reason: %q", resp.FinishReason)
		}
	}
}

func pdfToText(pdf []byte) ([]byte, error) {
	cmd := exec.Command("pdftotext", "-", "-")
	cmd.Stdin = bytes.NewReader(pdf)
	return cmd.Output()
}

func cleanText(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

// BOGUS: jsonschema MarshalJSON doesn't work unless we import it HERE, via this decl!!!
var _ jsonschema.Reflector
