import type { Anthropic } from "@anthropic-ai/sdk"
import { type ModelInfo, type VertexModelId, vertexDefaultModelId, vertexModels } from "@roo-code/types"

import type { ApiHandlerOptions } from "../../shared/api"

import { getModelParams } from "../transform/model-params"
import { convertAnthropicMessageToGemini } from "../transform/gemini-format"

import { GeminiHandler } from "./gemini"
import type { SingleCompletionHandler, ApiHandlerCreateMessageMetadata } from "../index"
import type { ApiStream, GroundingSource } from "../transform/stream"

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
			try {
				let hasAssistantContent = false
				for await (const chunk of this.createMessageExpress(systemInstruction, messages, metadata)) {
					if (chunk.type === "text" || chunk.type === "tool_call_partial" || chunk.type === "reasoning") {
						hasAssistantContent = true
					}
					yield chunk
				}

				if (!hasAssistantContent) {
					console.warn("Vertex Express: createMessage yielded no assistant content, yielding placeholder")
					yield { type: "text", text: " " }
				}
			} catch (error) {
				const errorMessage = error instanceof Error ? error.message : String(error)
				console.error(`Vertex Express creation error: ${errorMessage}`)

				// Propagate detailed error to prevent generic "Model Response Incomplete"
				if (errorMessage.includes("Finish Reason")) {
					yield { type: "text", text: `Error: ${errorMessage}` }
				}

				throw error
			}
			return
		}

		// Use the base GeminiHandler implementation for standard Vertex AI
		yield* super.createMessage(systemInstruction, messages, metadata)
	}

	/**
	 * Express mode using direct API calls with API key only.
	 * No GCP project or OAuth required.
	 * Documentation: https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/express-mode/api-reference
	 */
	private async *createMessageExpress(
		systemInstruction: string,
		messages: any[],
		metadata?: ApiHandlerCreateMessageMetadata,
	): ApiStream {
		const { id: model, info, maxTokens, reasoning: thinkingConfig } = this.getModel()
		const apiKey = this.options.vertexApiKey!

		// Reset per-request metadata
		this.lastThoughtSignature = undefined
		this.lastResponseId = undefined

		const includeThoughtSignatures = Boolean(thinkingConfig)

		// Filter reasoning messages
		const geminiMessages = messages.filter((message) => {
			const meta = message as { type?: string }
			return meta.type !== "reasoning"
		}) as Anthropic.Messages.MessageParam[]

		// Build tool map
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
			.map((content) => {
				// Vertex AI requires that functionResponse parts be sent with role: "function"
				if (content.parts?.some((part) => "functionResponse" in part)) {
					content.role = "function"
				}

				// Clean up functionResponse: Vertex AI rejects the inner 'name' field in the response object
				if (content.parts) {
					content.parts = content.parts.map((part: any) => {
						if (part.thoughtSignature) {
							const { thoughtSignature, ...rest } = part
							return { ...rest, thought_signature: thoughtSignature }
						}
						if (part.functionResponse) {
							const { functionResponse, ...rest } = part
							if (functionResponse.response?.name) {
								const { name, ...innerRest } = functionResponse.response
								return {
									...rest,
									function_response: {
										...functionResponse,
										response: innerRest,
									},
								}
							}
							return { ...rest, function_response: functionResponse }
						}
						return part
					})
				}
				return content
			})

		const tools: any[] = []
		if (metadata?.tools && metadata.tools.length > 0) {
			tools.push({
				function_declarations: metadata.tools.map((tool: any) => ({
					name: tool.function.name,
					description: tool.function.description,
					parameters: this.cleanSchema(tool.function.parameters),
				})),
			})
		} else {
			// Google built-in tools are mutually exclusive with function declarations
			if (this.options.enableUrlContext) {
				tools.push({ url_context: {} })
			}

			if (this.options.enableGrounding) {
				tools.push({ google_search_retrieval: {} })
			}
		}

		const cleanModelId = model.endsWith(":thinking") ? model.replace(":thinking", "") : model
		const url = `https://aiplatform.googleapis.com/v1/publishers/google/models/${cleanModelId}:streamGenerateContent?key=${apiKey}`

		// Convert thinkingConfig from camelCase (SDK format) to snake_case (REST API format)
		const thinkingConfigSnakeCase = thinkingConfig
			? {
					...(thinkingConfig.thinkingBudget !== undefined
						? { thinking_budget: thinkingConfig.thinkingBudget }
						: {}),
					...(thinkingConfig.thinkingLevel !== undefined
						? { thinking_level: thinkingConfig.thinkingLevel }
						: {}),
					...(thinkingConfig.includeThoughts !== undefined
						? { include_thoughts: thinkingConfig.includeThoughts }
						: {}),
				}
			: undefined

		const body = {
			system_instruction: { parts: [{ text: systemInstruction }] },
			contents: googleMessages,
			tools: tools.length > 0 ? tools : undefined,
			generation_config: {
				temperature: this.options.modelTemperature ?? info.defaultTemperature ?? 1.0,
				max_output_tokens: this.options.modelMaxTokens ?? maxTokens ?? 8192,
				...(thinkingConfigSnakeCase ? { thinking_config: thinkingConfigSnakeCase } : {}),
			},
			safety_settings: [
				{ category: "HARM_CATEGORY_DANGEROUS_CONTENT", threshold: "BLOCK_NONE" },
				{ category: "HARM_CATEGORY_HATE_SPEECH", threshold: "BLOCK_NONE" },
				{ category: "HARM_CATEGORY_HARASSMENT", threshold: "BLOCK_NONE" },
				{ category: "HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold: "BLOCK_NONE" },
				{ category: "HARM_CATEGORY_CIVIC_INTEGRITY", threshold: "BLOCK_NONE" },
			],
		}

		console.log(`Vertex Express Request URL: ${url}`)
		console.log(`Vertex Express Request Body: ${JSON.stringify(body, null, 2)}`)

		const response = await fetch(url, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify(body),
		})

		if (!response.ok) {
			const errorText = await response.text()
			console.error(`Vertex Express HTTP Error (${response.status}): ${errorText}`)
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
			let pendingGroundingMetadata: any | undefined
			let hasYieldedAssistantContent = false

			while (true) {
				const { done, value } = await reader.read()
				if (done) break

				const decodedChunk = decoder.decode(value, { stream: true })
				buffer += decodedChunk

				while (cursor < buffer.length) {
					const char = buffer[cursor]
					if (inString) {
						if (char === "\\") escaped = !escaped
						else if (char === '"' && !escaped) inString = false
						else escaped = false
					} else {
						if (char === '"') inString = true
						else if (char === "{") {
							if (depth === 0) startIndex = cursor
							depth++
						} else if (char === "}") {
							depth--
							if (depth === 0 && startIndex !== -1) {
								const jsonStr = buffer.substring(startIndex, cursor + 1)
								try {
									const chunk = JSON.parse(jsonStr)
									// Log raw chunk for debugging
									console.log(`Vertex Express Chunk: ${JSON.stringify(chunk)}`)

									if (chunk.responseId) this.lastResponseId = chunk.responseId

									const candidate = chunk.candidates?.[0]
									if (candidate) {
										if (candidate.groundingMetadata)
											pendingGroundingMetadata = candidate.groundingMetadata

										if (candidate.finishReason && candidate.finishReason !== "STOP") {
											const reason = candidate.finishReason
											if (["SAFETY", "RECITATION", "OTHER"].includes(reason)) {
												throw new Error(`Vertex Express Finish Reason: ${reason}`)
											}
										}

										if (candidate.content?.parts) {
											for (const part of candidate.content.parts) {
												if (
													(part.thought_signature || part.thoughtSignature) &&
													thinkingConfig
												) {
													this.lastThoughtSignature =
														part.thought_signature || part.thoughtSignature
												}

												// Function calls should never be treated as reasoning, even if they carry a thoughtSignature
												if (part.functionCall) {
													hasYieldedAssistantContent = true
													const callId = `${part.functionCall.name}-${toolCallCounter}`
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
														arguments: JSON.stringify(part.functionCall.args),
													}
													toolCallCounter++
													continue
												}

												// Treat reasoning blocks as content
												// Note: 'thought' flag and 'thoughtSignature' can come in the same part or separate ones.
												if (part.thought === true || part.role === "thought") {
													if (part.text) {
														hasYieldedAssistantContent = true
														yield { type: "reasoning", text: part.text }
													} else {
														// Structural reasoning parts mark as having content
														hasYieldedAssistantContent = true
														// Yield an empty reasoning block to ensure the UI sees activity
														yield { type: "reasoning", text: "" }
													}
													continue
												}

												if (part.text) {
													hasYieldedAssistantContent = true
													const parts = part.text.split(
														/(<think(?:\s.*?)?>|<\/think(?:\s.*?)?>)/gi,
													)
													for (const p of parts) {
														const lowerP = p.toLowerCase()
														if (lowerP.startsWith("<think")) inThought = true
														else if (lowerP.startsWith("</think")) inThought = false
														else if (p.length > 0)
															yield { type: inThought ? "reasoning" : "text", text: p }
													}
												} else if (part.functionCall) {
													hasYieldedAssistantContent = true
													const callId = `${part.functionCall.name}-${toolCallCounter}`
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
														arguments: JSON.stringify(part.functionCall.args),
													}
													toolCallCounter++
												}
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
											totalCost: this.calculateCost({ info, inputTokens, outputTokens }),
										}
									}
								} catch (e) {
									if (e instanceof Error && e.message.startsWith("Vertex Express Finish Reason"))
										throw e
									console.error("JSON Parse Error:", e)
								}
								buffer = buffer.substring(cursor + 1)
								cursor = -1
								startIndex = -1
							}
						}
					}
					cursor++
				}
			}

			if (!hasYieldedAssistantContent) {
				console.warn("Vertex Express: No assistant content yielded in stream, yielding placeholder")
				yield { type: "text", text: " " }
			}

			if (pendingGroundingMetadata) {
				const sources = this.extractGroundingSources(pendingGroundingMetadata)
				if (sources.length > 0) yield { type: "grounding", sources }
			}
		} finally {
			reader.releaseLock()
		}
	}

	override getModel() {
		const modelId = this.options.apiModelId
		const id = modelId && modelId in vertexModels ? (modelId as VertexModelId) : vertexDefaultModelId
		let info: ModelInfo = vertexModels[id]
		const params = getModelParams({
			format: "gemini",
			modelId: id,
			model: info,
			settings: this.options,
			defaultTemperature: info.defaultTemperature ?? 1,
		})

		info = {
			...info,
			excludedTools: [...new Set([...(info.excludedTools || []), "apply_diff"])],
			includedTools: [...new Set([...(info.includedTools || []), "edit"])],
		}

		return { id: id.endsWith(":thinking") ? id.replace(":thinking", "") : id, info, ...params }
	}

	protected override extractGroundingSources(groundingMetadata: any): GroundingSource[] {
		const chunks = groundingMetadata?.groundingChunks
		if (!chunks) return []
		return chunks
			.map((chunk: any): GroundingSource | null => {
				const uri = chunk.web?.uri
				const title = chunk.web?.title || uri || "Unknown Source"
				return uri ? { title, url: uri } : null
			})
			.filter((source: any): source is GroundingSource => source !== null)
	}

	public override getResponseId(): string | undefined {
		return this.lastResponseId
	}

	public override getThoughtSignature(): string | undefined {
		return this.lastThoughtSignature
	}

	private cleanSchema(schema: any): any {
		if (!schema || typeof schema !== "object") return schema
		if (Array.isArray(schema)) return schema.map((item) => this.cleanSchema(item))
		const out: any = {}
		for (const key in schema) {
			if (key === "properties") {
				out[key] = {}
				for (const propertyKey in schema[key])
					out[key][propertyKey] = this.cleanSchema(schema[key][propertyKey])
				continue
			}
			if (
				[
					"exclusiveMinimum",
					"exclusiveMaximum",
					"minimum",
					"maximum",
					"multipleOf",
					"minLength",
					"maxLength",
					"minItems",
					"maxItems",
					"uniqueItems",
					"pattern",
					"const",
					"additionalProperties",
					"title",
					"default",
					"examples",
					"$schema",
					"$id",
					"unevaluatedProperties",
					"propertyNames",
					"minProperties",
					"maxProperties",
					"allOf",
					"oneOf",
					"anyOf",
					"not",
					"if",
					"then",
					"else",
					"dependentRequired",
					"dependentSchemas",
				].includes(key)
			)
				continue
			if (key === "type" && Array.isArray(schema[key])) {
				const firstType = schema[key].find((t: string) => t !== "null")
				out[key] = firstType || schema[key][0]
				if (schema[key].includes("null")) out["nullable"] = true
				continue
			}
			out[key] = this.cleanSchema(schema[key])
		}
		return out
	}
}
