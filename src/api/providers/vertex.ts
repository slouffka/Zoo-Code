import { type ModelInfo, type VertexModelId, vertexDefaultModelId, vertexModels } from "@roo-code/types"

import type { ApiHandlerOptions } from "../../shared/api"

import { getModelParams } from "../transform/model-params"

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
			yield* this.createMessageExpress(systemInstruction, messages)
			return
		}

		// Use the base GeminiHandler implementation for standard Vertex AI
		yield* super.createMessage(systemInstruction, messages, metadata)
	}

	/**
	 * Express mode using direct API calls with API key only.
	 * No GCP project or OAuth required.
	 */
	private async *createMessageExpress(systemInstruction: string, messages: any[]): ApiStream {
		const modelId = this.options.apiModelId || vertexDefaultModelId
		const apiKey = this.options.vertexApiKey!

		const googleMessages = messages.map((msg: any) => ({
			role: msg.role === "assistant" ? "model" : "user",
			parts: [{ text: typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content) }],
		}))

		// Handle specific model suffixes if present (e.g. :thinking)
		const cleanModelId = modelId.endsWith(":thinking") ? modelId.replace(":thinking", "") : modelId
		const url = `https://aiplatform.googleapis.com/v1beta1/publishers/google/models/${cleanModelId}:streamGenerateContent?key=${apiKey}`

		const response = await fetch(url, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				systemInstruction: { parts: [{ text: systemInstruction }] },
				contents: googleMessages,
				generationConfig: {
					temperature: this.options.modelTemperature ?? 0.0,
					maxOutputTokens: this.options.modelMaxTokens ?? 8192,
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
			// State machine variables for JSON parsing
			let inString = false
			let escaped = false
			let depth = 0
			let startIndex = -1

			while (true) {
				const { done, value } = await reader.read()
				if (done) break

				const chunkStr = decoder.decode(value, { stream: true })
				buffer += chunkStr

				// Reset state for current buffer processing
				inString = false
				escaped = false
				depth = 0
				startIndex = -1
				let lastProcessedIndex = -1

				for (let i = 0; i < buffer.length; i++) {
					const char = buffer[i]

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
							if (depth === 0) startIndex = i
							depth++
						} else if (char === "}") {
							depth--
							if (depth === 0 && startIndex !== -1) {
								// Complete JSON object found
								const jsonStr = buffer.substring(startIndex, i + 1)
								try {
									const chunk = JSON.parse(jsonStr)

									const text = chunk.candidates?.[0]?.content?.parts?.[0]?.text
									if (text) yield { type: "text", text }

									if (chunk.usageMetadata) {
										yield {
											type: "usage",
											inputTokens: chunk.usageMetadata.promptTokenCount,
											outputTokens: chunk.usageMetadata.candidatesTokenCount,
										}
									}
								} catch (e) {
									console.error("JSON Parse Error:", e)
								}

								lastProcessedIndex = i
								startIndex = -1
							}
						}
					}
				}

				// Remove processed data from buffer
				if (lastProcessedIndex !== -1) {
					buffer = buffer.substring(lastProcessedIndex + 1)
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
}
