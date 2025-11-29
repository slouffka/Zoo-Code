import type { Anthropic } from "@anthropic-ai/sdk"
import { type ModelInfo, type VertexModelId, vertexDefaultModelId, vertexModels } from "@roo-code/types"

import type { ApiHandlerOptions } from "../../shared/api"

import { getModelParams } from "../transform/model-params"
import { convertAnthropicMessageToGemini } from "../transform/gemini-format"

import { GeminiHandler } from "./gemini"
import type { SingleCompletionHandler, ApiHandlerCreateMessageMetadata } from "../index"
import type { ApiStream } from "../transform/stream"

/**
 * Vertex AI provider.
 * Inherits from GeminiHandler to reuse its logic, as Vertex AI (Express and Standard)
 * now uses the same underlying Gemini API structure.
 */
export class VertexHandler extends GeminiHandler implements SingleCompletionHandler {
	constructor(options: ApiHandlerOptions) {
		super({ ...options, isVertex: true })
	}

	override async *createMessage(
		systemInstruction: string,
		messages: any[],
		metadata?: ApiHandlerCreateMessageMetadata,
	): ApiStream {
		// Vertex AI Express mode (API Key only)
		if (this.options.vertexApiKey) {
			yield* this.createMessageExpress(systemInstruction, messages, metadata)
			return
		}

		// Use the base GeminiHandler implementation for standard Vertex AI
		yield* super.createMessage(systemInstruction, messages, metadata)
	}

	/**
	 * Express mode using direct API calls with API key only.
	 * No GCP project or OAuth required.
	 */
	private async *createMessageExpress(
		systemInstruction: string,
		messages: any[],
		metadata?: ApiHandlerCreateMessageMetadata,
	): ApiStream {
		const { id: model, info, maxTokens, reasoning: thinkingConfig } = this.getModel()
		const apiKey = this.options.vertexApiKey!

		// Only forward encrypted reasoning continuations (thoughtSignature) when we are
		// using reasoning (thinkingConfig is present). Both effort-based (thinkingLevel)
		// and budget-based (thinkingBudget) models require this for active loops.
		const includeThoughtSignatures = Boolean(thinkingConfig)

		// Filter out "reasoning" meta messages that are not valid Anthropic messages
		const geminiMessages = messages.filter((message) => {
			const meta = message as { type?: string }
			return meta.type !== "reasoning"
		}) as Anthropic.Messages.MessageParam[]

		// Build a map of tool IDs to names from previous messages
		const toolIdToName = new Map<string, string>()
		for (const message of messages) {
			if (Array.isArray(message.content)) {
				for (const block of message.content) {
					if (block.type === "tool_use") {
						toolIdToName.set(block.id, block.name)
					}
				}
			}
		}

		const googleMessages = geminiMessages
			.map((message) => convertAnthropicMessageToGemini(message, { includeThoughtSignatures, toolIdToName }))
			.flat()

		// Build tools for Gemini API
		const tools: any[] = []
		if (metadata?.tools && metadata.tools.length > 0) {
			tools.push({
				functionDeclarations: metadata.tools.map((tool: any) => ({
					name: tool.function.name,
					description: tool.function.description,
					parameters: this.cleanSchema(tool.function.parameters),
				})),
			})
		}

		// Handle specific model suffixes if present (e.g. :thinking)
		const cleanModelId = model.endsWith(":thinking") ? model.replace(":thinking", "") : model
		const url = `https://aiplatform.googleapis.com/v1beta1/publishers/google/models/${cleanModelId}:streamGenerateContent?key=${apiKey}`

		const response = await fetch(url, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				systemInstruction: { parts: [{ text: systemInstruction }] },
				contents: googleMessages,
				tools: tools.length > 0 ? tools : undefined,
				generationConfig: {
					temperature: this.options.modelTemperature ?? info.defaultTemperature ?? 1.0,
					maxOutputTokens: this.options.modelMaxTokens ?? maxTokens ?? 8192,
					...(thinkingConfig ? { thinkingConfig } : {}),
				},
				safetySettings: [
					{ category: "HARM_CATEGORY_DANGEROUS_CONTENT", threshold: "BLOCK_ONLY_HIGH" },
					{ category: "HARM_CATEGORY_HATE_SPEECH", threshold: "BLOCK_ONLY_HIGH" },
					{ category: "HARM_CATEGORY_HARASSMENT", threshold: "BLOCK_ONLY_HIGH" },
					{ category: "HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold: "BLOCK_ONLY_HIGH" },
				],
			}),
		})

		if (!response.ok) {
			const errorText = await response.text()
			throw new Error(`Vertex Express Error (${response.status}): ${errorText}`)
		}
		if (!response.body) throw new Error("No response body")

		const reader = response.body.getReader()
		const decoder = new TextDecoder()
		let buffer = ""

		try {
			let inThought = false
			let inString = false
			let escaped = false
			let depth = 0
			let startIndex = -1
			let cursor = 0
			let toolCallCounter = 0

			while (true) {
				const { done, value } = await reader.read()
				if (done) break

				const chunkStr = decoder.decode(value, { stream: true })
				buffer += chunkStr

				while (cursor < buffer.length) {
					const char = buffer[cursor]

					if (inString) {
						if (char === "\\") {
							escaped = !escaped
						} else if (char === '"' && !escaped) {
							inString = false
						} else {
							escaped = false
						}
					} else {
						if (char === '"') {
							inString = true
						} else if (char === "{") {
							if (depth === 0) startIndex = cursor
							depth++
						} else if (char === "}") {
							depth--
							if (depth === 0 && startIndex !== -1) {
								// Complete JSON object found
								const jsonStr = buffer.substring(startIndex, cursor + 1)
								try {
									const chunk = JSON.parse(jsonStr)

									const candidate = chunk.candidates?.[0]
									if (candidate?.content?.parts) {
										for (const part of candidate.content.parts) {
											// Handle standard Gemini thought field if present
											if (part.thought) {
												if (part.text) {
													yield { type: "reasoning", text: part.text }
												}
												continue
											}

											if (part.text) {
												const parts = part.text.split(
													/(<think(?:\s.*?)?>|<\/think(?:\s.*?)?>)/gi,
												)
												for (const p of parts) {
													const lowerP = p.toLowerCase()
													if (lowerP.startsWith("<think")) {
														inThought = true
													} else if (lowerP.startsWith("</think")) {
														inThought = false
													} else if (p.length > 0) {
														yield { type: inThought ? "reasoning" : "text", text: p }
													}
												}
											} else if (part.functionCall) {
												const callId = `${part.functionCall.name}-${toolCallCounter}`
												const args = JSON.stringify(part.functionCall.args)

												yield {
													type: "tool_call_partial",
													index: toolCallCounter,
													id: callId,
													name: part.functionCall.name,
													arguments: undefined,
												}

												yield {
													type: "tool_call_partial",
													index: toolCallCounter,
													id: callId,
													name: undefined,
													arguments: args,
												}

												toolCallCounter++
											}
										}
									}

									const usage = chunk.usageMetadata || chunk.usage_metadata
									if (usage) {
										const inputTokens = usage.promptTokenCount ?? usage.prompt_token_count ?? 0
										const outputTokens =
											usage.candidatesTokenCount ?? usage.candidates_token_count ?? 0
										yield {
											type: "usage",
											inputTokens,
											outputTokens,
											totalCost: this.calculateCost({
												info,
												inputTokens,
												outputTokens,
											}),
										}
									}
								} catch (e) {
									console.error("JSON Parse Error:", e)
								}

								// Remove processed data from buffer
								buffer = buffer.substring(cursor + 1)
								cursor = -1 // will be incremented to 0
								startIndex = -1
							}
						}
					}
					cursor++
				}
			}
		} finally {
			reader.releaseLock()
		}
	}

	override getModel() {
		const modelId = this.options.apiModelId
		let id = modelId && modelId in vertexModels ? (modelId as VertexModelId) : vertexDefaultModelId
		let info: ModelInfo = vertexModels[id]
		const params = getModelParams({
			format: "gemini",
			modelId: id,
			model: info,
			settings: this.options,
			defaultTemperature: info.defaultTemperature ?? 1,
		})

		// Vertex Gemini models perform better with the edit tool instead of apply_diff.
		info = {
			...info,
			excludedTools: [...new Set([...(info.excludedTools || []), "apply_diff"])],
			includedTools: [...new Set([...(info.includedTools || []), "edit"])],
		}

		// The `:thinking` suffix indicates that the model is a "Hybrid"
		// reasoning model and that reasoning is required to be enabled.
		// The actual model ID honored by Gemini's API does not have this
		// suffix.
		return { id: id.endsWith(":thinking") ? id.replace(":thinking", "") : id, info, ...params }
	}

	/**
	 * Removes unsupported JSON schema keywords from the tool definition.
	 * Vertex AI's Function Calling API is strict and rejects requests containing
	 * standard JSON schema fields like "additionalProperties", "default",
	 * and validation keywords (min/max/pattern) that are not supported in its Protobuf definition.
	 */
	private cleanSchema(schema: any): any {
		if (!schema || typeof schema !== "object") return schema

		if (Array.isArray(schema)) {
			return schema.map((item) => this.cleanSchema(item))
		}

		const out: any = {}
		for (const key in schema) {
			if (
				key === "exclusiveMinimum" ||
				key === "exclusiveMaximum" ||
				key === "minimum" ||
				key === "maximum" ||
				key === "multipleOf" ||
				key === "minLength" ||
				key === "maxLength" ||
				key === "pattern" ||
				key === "additionalProperties" ||
				key === "title" ||
				key === "default" ||
				key === "examples" ||
				key === "$schema" ||
				key === "$id"
			) {
				continue
			}

			if (key === "type" && Array.isArray(schema[key])) {
				// Vertex AI doesn't support array types (e.g. ["string", "null"])
				// Use the first non-null type
				const firstType = schema[key].find((t: string) => t !== "null")
				out[key] = firstType || schema[key][0]
				if (schema[key].includes("null")) {
					out["nullable"] = true
				}
				continue
			}

			out[key] = this.cleanSchema(schema[key])
		}
		return out
	}
}
