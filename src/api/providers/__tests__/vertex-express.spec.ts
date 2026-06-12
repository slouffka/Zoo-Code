import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { ApiHandlerOptions } from "../../../shared/api"
import { VertexHandler } from "../vertex"

// Mock i18next
vi.mock("i18next", () => ({
	t: (key: string) => key,
}))

describe("VertexHandler Express Mode", () => {
	let handler: VertexHandler
	let fetchMock: any

	const options: ApiHandlerOptions = {
		apiModelId: "gemini-2.0-flash-001",
		vertexApiKey: "test-api-key",
		vertexProjectId: undefined,
		vertexRegion: undefined,
	}

	beforeEach(() => {
		fetchMock = vi.fn()
		global.fetch = fetchMock
		handler = new VertexHandler(options)
	})

	afterEach(() => {
		vi.restoreAllMocks()
	})

	it("should use correct generation config with defaults", async () => {
		const stream = new ReadableStream({
			start(controller) {
				controller.close()
			},
		})
		const response = {
			ok: true,
			body: stream,
			text: () => Promise.resolve(""),
		}
		fetchMock.mockResolvedValue(response)

		const generator = handler.createMessage("system prompt", [{ role: "user", content: "hello" }])
		await generator.next()

		expect(fetchMock).toHaveBeenCalledWith(
			expect.stringContaining("/v1/publishers/google/models/"),
			expect.objectContaining({
				method: "POST",
			}),
		)

		const callArgs = JSON.parse(fetchMock.mock.calls[0][1].body)
		expect(callArgs.generation_config.temperature).toBe(1)
		expect(callArgs.generation_config.max_output_tokens).toBe(8192)
		expect(callArgs.system_instruction).toBeDefined()
	})

	it("should use user provided configuration", async () => {
		const customOptions = {
			...options,
			modelTemperature: 0.5,
			modelMaxTokens: 1000,
		}
		handler = new VertexHandler(customOptions)

		const stream = new ReadableStream({
			start(controller) {
				controller.close()
			},
		})
		const response = {
			ok: true,
			body: stream,
			text: () => Promise.resolve(""),
		}
		fetchMock.mockResolvedValue(response)

		const generator = handler.createMessage("system prompt", [{ role: "user", content: "hello" }])
		await generator.next()

		const callArgs = JSON.parse(fetchMock.mock.calls[0][1].body)
		expect(callArgs.generation_config.temperature).toBe(0.5)
		expect(callArgs.generation_config.max_output_tokens).toBe(1000)
	})

	it("should correctly format complex messages using convertAnthropicMessageToGemini", async () => {
		const stream = new ReadableStream({
			start(controller) {
				controller.close()
			},
		})
		const response = {
			ok: true,
			body: stream,
			text: () => Promise.resolve(""),
		}
		fetchMock.mockResolvedValue(response)

		// Create a message with tool use structure that would have been JSON-stringified naively
		const messages = [
			{
				role: "assistant",
				content: [
					{ type: "text", text: "I will run this." },
					{
						type: "tool_use",
						id: "call_1",
						name: "execute_command",
						input: { command: "ls" },
					},
				],
			},
		]

		// Mock toolIdToName logic if needed, but here we just want to see the request body structure
		const generator = handler.createMessage("system prompt", messages as any)
		await generator.next()

		const callArgs = JSON.parse(fetchMock.mock.calls[0][1].body)
		const contents = callArgs.contents

		// Should be a proper structure, NOT a JSON string in 'text'
		expect(contents[0].parts[0]).toHaveProperty("text", "I will run this.")
		expect(contents[0].parts[1]).toHaveProperty("functionCall")
		expect(contents[0].parts[1].functionCall).toEqual({
			name: "execute_command",
			args: { command: "ls" },
		})
	})

	it("should parse streaming JSON correctly", async () => {
		const chunks = [
			'{"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}',
			'{"candidates": [{"content": {"parts": [{"text": " world"}]}}]}',
			'{"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5}}',
		]

		const encoder = new TextEncoder()
		const stream = new ReadableStream({
			start(controller) {
				chunks.forEach((chunk) => controller.enqueue(encoder.encode(chunk)))
				controller.close()
			},
		})

		fetchMock.mockResolvedValue({
			ok: true,
			body: stream,
			text: () => Promise.resolve(""),
		})

		const messages = []
		for await (const msg of handler.createMessage("sys", [])) {
			messages.push(msg)
		}

		expect(messages).toEqual([
			{ type: "text", text: "Hello" },
			{ type: "text", text: " world" },
			{
				type: "usage",
				inputTokens: 10,
				outputTokens: 5,
				totalCost: expect.any(Number),
			},
		])
	})

	it("should handle split JSON chunks", async () => {
		const chunks = [
			'{"candidates": [{"content": {"parts": [{"text": "Split"}]}}',
			"]}",
			'{"candidates": [{"content": {"parts": [{"text": " "}]}}]}',
			'{"candidates": [{"content": {"parts": [{"text": "Chunk"}]}}]}',
		]

		const encoder = new TextEncoder()
		const stream = new ReadableStream({
			start(controller) {
				chunks.forEach((chunk) => controller.enqueue(encoder.encode(chunk)))
				controller.close()
			},
		})

		fetchMock.mockResolvedValue({
			ok: true,
			body: stream,
			text: () => Promise.resolve(""),
		})

		const messages = []
		for await (const msg of handler.createMessage("sys", [])) {
			messages.push(msg)
		}

		expect(messages).toEqual([
			{ type: "text", text: "Split" },
			{ type: "text", text: " " },
			{ type: "text", text: "Chunk" },
		])
	})

	it("should parse <think> tags and emit reasoning events", async () => {
		const chunks = [
			'{"candidates": [{"content": {"parts": [{"text": "<think>This is a thought"}]}}]}',
			'{"candidates": [{"content": {"parts": [{"text": " still thinking</think>"}]}}]}',
			'{"candidates": [{"content": {"parts": [{"text": "This is the response"}]}}]}',
		]

		const encoder = new TextEncoder()
		const stream = new ReadableStream({
			start(controller) {
				chunks.forEach((chunk) => controller.enqueue(encoder.encode(chunk)))
				controller.close()
			},
		})

		fetchMock.mockResolvedValue({
			ok: true,
			body: stream,
			text: () => Promise.resolve(""),
		})

		const messages = []
		for await (const msg of handler.createMessage("sys", [])) {
			messages.push(msg)
		}

		expect(messages).toEqual([
			{ type: "reasoning", text: "This is a thought" },
			{ type: "reasoning", text: " still thinking" },
			{ type: "text", text: "This is the response" },
		])
	})

	it("should include tools in request and clean schema parameters", async () => {
		const stream = new ReadableStream({
			start(controller) {
				controller.close()
			},
		})
		const response = {
			ok: true,
			body: stream,
			text: () => Promise.resolve(""),
		}
		fetchMock.mockResolvedValue(response)

		const tools = [
			{
				name: "test_tool",
				description: "A test tool",
				parameters: {
					type: "object",
					properties: {
						param1: {
							type: "string",
							description: "Parameter 1",
							default: "default_value",
						},
						param2: {
							type: "integer",
							minimum: 0,
							maximum: 10,
							exclusiveMaximum: true,
							const: 5,
						},
						param3: {
							type: ["string", "null"],
							description: "A nullable string",
							minItems: 1,
							allOf: [{ type: "string" }],
						},
					},
					required: ["param1"],
					additionalProperties: false,
				},
			},
		]

		// Pass tools in metadata
		const generator = handler.createMessage("system prompt", [], {
			tools: tools.map((t) => ({
				type: "function",
				function: {
					name: t.name,
					description: t.description,
					parameters: t.parameters,
				},
			})),
			taskId: "test-task-id",
		})
		await generator.next()

		const callArgs = JSON.parse(fetchMock.mock.calls[0][1].body)
		const sentTools = callArgs.tools

		expect(sentTools).toHaveLength(1)
		expect(sentTools[0].function_declarations).toHaveLength(1)

		const decl = sentTools[0].function_declarations[0]
		expect(decl.name).toBe("test_tool")
		expect(decl.description).toBe("A test tool")

		// Verify schema cleaning
		const params = decl.parameters
		expect(params).not.toHaveProperty("additionalProperties")
		expect(params.properties.param1).not.toHaveProperty("default")
		expect(params.properties.param2).not.toHaveProperty("exclusiveMaximum")
		expect(params.properties.param2).not.toHaveProperty("minimum")
		expect(params.properties.param2).not.toHaveProperty("maximum")
		expect(params.properties.param2).not.toHaveProperty("const")
		expect(params.properties.param3).not.toHaveProperty("minItems")
		expect(params.properties.param3).not.toHaveProperty("allOf")

		// Verify preserved fields
		expect(params.properties.param1.type).toBe("string")
		expect(params.properties.param1.description).toBe("Parameter 1")
		expect(params.properties.param2.type).toBe("integer")
		expect(params.required).toEqual(["param1"])

		// Verify array type flattening (nullable handling)
		expect(params.properties.param3.type).toBe("string")
		expect(params.properties.param3.nullable).toBe(true)
	})

	it("should clean up functionResponse structure for Vertex AI", async () => {
		const stream = new ReadableStream({
			start(controller) {
				controller.close()
			},
		})
		const response = {
			ok: true,
			body: stream,
			text: () => Promise.resolve(""),
		}
		fetchMock.mockResolvedValue(response)

		// Mock a tool_use followed by a tool_result
		const messages = [
			{
				role: "assistant",
				content: [
					{
						type: "tool_use",
						id: "call_1",
						name: "read_file",
						input: { path: "test.ts" },
					},
				],
			},
			{
				role: "user",
				content: [
					{
						type: "tool_result",
						tool_use_id: "call_1",
						content: "file content",
					},
				],
			},
		]

		const generator = handler.createMessage("system prompt", messages as any)
		await generator.next()

		const callArgs = JSON.parse(fetchMock.mock.calls[0][1].body)
		const contents = callArgs.contents

		// Verify the functionResponse part
		const lastMessage = contents[contents.length - 1]
		expect(lastMessage.role).toBe("function")
		const functionResponsePart = lastMessage.parts[0]

		expect(functionResponsePart).toHaveProperty("function_response")
		expect(functionResponsePart.function_response).toHaveProperty("name", "read_file")
		expect(functionResponsePart.function_response.response).toHaveProperty("content", "file content")
		// The key fix: confirm 'name' is NOT in the inner response object
		expect(functionResponsePart.function_response.response).not.toHaveProperty("name")
	})

	it("should set role to 'function' for messages with functionResponse", async () => {
		const stream = new ReadableStream({
			start(controller) {
				controller.close()
			},
		})
		fetchMock.mockResolvedValue({
			ok: true,
			body: stream,
			text: () => Promise.resolve(""),
		})

		const messages = [
			{
				role: "assistant",
				content: [
					{
						type: "tool_use",
						id: "call_1",
						name: "read_file",
						input: { path: "test.ts" },
					},
				],
			},
			{
				role: "user",
				content: [{ type: "tool_result", tool_use_id: "call_1", content: "result" }],
			},
		]

		const generator = handler.createMessage("system prompt", messages as any)
		await generator.next()

		const callArgs = JSON.parse(fetchMock.mock.calls[0][1].body)
		const lastMessage = callArgs.contents[callArgs.contents.length - 1]

		expect(lastMessage.role).toBe("function")
		expect(lastMessage.parts[0]).toHaveProperty("function_response")
	})

	it("should correctly handle apply_patch tool schema", async () => {
		const stream = new ReadableStream({
			start(controller) {
				controller.close()
			},
		})
		const response = {
			ok: true,
			body: stream,
			text: () => Promise.resolve(""),
		}
		fetchMock.mockResolvedValue(response)

		const applyPatchTool = {
			name: "apply_patch",
			description: "Apply patches to files.",
			parameters: {
				type: "object",
				properties: {
					patch: {
						type: "string",
						description: "The complete patch text.",
					},
				},
				required: ["patch"],
				additionalProperties: false,
			},
		}

		const generator = handler.createMessage("system prompt", [], {
			tools: [
				{
					type: "function",
					function: applyPatchTool,
				},
			],
			taskId: "test-task-id",
		})
		await generator.next()

		const callArgs = JSON.parse(fetchMock.mock.calls[0][1].body)
		const sentTools = callArgs.tools

		expect(sentTools).toHaveLength(1)
		expect(sentTools[0].function_declarations).toHaveLength(1)

		const decl = sentTools[0].function_declarations[0]
		expect(decl.name).toBe("apply_patch")
		expect(decl.description).toBe("Apply patches to files.")

		const params = decl.parameters
		expect(params).not.toHaveProperty("additionalProperties")
		expect(params.properties.patch.type).toBe("string")
		expect(params.properties.patch.description).toBe("The complete patch text.")
		expect(params.required).toEqual(["patch"])
	})

	it("should preserve properties named 'title' in tool schema", async () => {
		const stream = new ReadableStream({
			start(controller) {
				controller.close()
			},
		})
		const response = {
			ok: true,
			body: stream,
			text: () => Promise.resolve(""),
		}
		fetchMock.mockResolvedValue(response)

		const tools = [
			{
				name: "create_task",
				description: "Create a task",
				parameters: {
					type: "object",
					properties: {
						title: {
							type: "string",
							description: "The title of the task",
						},
						description: {
							type: "string",
						},
					},
					required: ["title"],
				},
			},
		]

		const generator = handler.createMessage("system prompt", [], {
			tools: tools.map((t) => ({
				type: "function",
				function: {
					name: t.name,
					description: t.description,
					parameters: t.parameters,
				},
			})),
			taskId: "test-task-id",
		})
		await generator.next()

		const callArgs = JSON.parse(fetchMock.mock.calls[0][1].body)
		const sentTools = callArgs.tools
		const decl = sentTools[0].function_declarations[0]
		const params = decl.parameters

		expect(params.properties).toHaveProperty("title")
		expect(params.properties.title.type).toBe("string")
		expect(params.required).toContain("title")
	})

	it("should add google_search_retrieval tool when enableGrounding is true and no tools are present", async () => {
		const stream = new ReadableStream({
			start(controller) {
				controller.close()
			},
		})
		const response = {
			ok: true,
			body: stream,
			text: () => Promise.resolve(""),
		}
		fetchMock.mockResolvedValue(response)

		// Enable grounding in options
		const groundingHandler = new VertexHandler({
			...options,
			enableGrounding: true,
		})

		const generator = groundingHandler.createMessage("system prompt", [])
		await generator.next()

		const callArgs = JSON.parse(fetchMock.mock.calls[0][1].body)
		const sentTools = callArgs.tools

		expect(sentTools).toHaveLength(1)
		expect(sentTools[0]).toHaveProperty("google_search_retrieval")
	})

	it("should add url_context tool when enableUrlContext is true and no tools are present", async () => {
		const stream = new ReadableStream({
			start(controller) {
				controller.close()
			},
		})
		const response = {
			ok: true,
			body: stream,
			text: () => Promise.resolve(""),
		}
		fetchMock.mockResolvedValue(response)

		// Enable URL context in options
		const urlContextHandler = new VertexHandler({
			...options,
			enableUrlContext: true,
		})

		const generator = urlContextHandler.createMessage("system prompt", [])
		await generator.next()

		const callArgs = JSON.parse(fetchMock.mock.calls[0][1].body)
		const sentTools = callArgs.tools

		expect(sentTools).toHaveLength(1)
		expect(sentTools[0]).toHaveProperty("url_context")
	})

	it("should capture responseId and thought_signature from chunks", async () => {
		const chunks = [
			JSON.stringify({
				responseId: "test-response-id",
				candidates: [
					{
						content: {
							parts: [{ text: "Hello", thought_signature: "test-thought-sig" }],
						},
					},
				],
			}),
		]

		const encoder = new TextEncoder()
		const stream = new ReadableStream({
			start(controller) {
				chunks.forEach((chunk) => controller.enqueue(encoder.encode(chunk)))
				controller.close()
			},
		})

		fetchMock.mockResolvedValue({
			ok: true,
			body: stream,
			text: () => Promise.resolve(""),
		})

		// Use model with reasoning to enable thoughtSignature capture
		const reasoningHandler = new VertexHandler({
			...options,
			apiModelId: "gemini-2.5-flash-preview-05-20:thinking",
		})

		for await (const msg of reasoningHandler.createMessage("sys", [])) {
			// consume stream
		}

		expect(reasoningHandler.getResponseId()).toBe("test-response-id")
		expect(reasoningHandler.getThoughtSignature()).toBe("test-thought-sig")
	})

	it("should not treat functionCall parts as reasoning even when they have thoughtSignature", async () => {
		const chunks = [
			JSON.stringify({
				candidates: [
					{
						content: {
							role: "model",
							parts: [
								{
									functionCall: {
										name: "update_todo_list",
										args: { todos: "[ ] Task 1" },
									},
									thoughtSignature: "sig-123",
								},
							],
						},
					},
				],
				usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 5 },
			}),
		]

		const encoder = new TextEncoder()
		const stream = new ReadableStream({
			start(controller) {
				chunks.forEach((chunk) => controller.enqueue(encoder.encode(chunk)))
				controller.close()
			},
		})

		fetchMock.mockResolvedValue({
			ok: true,
			body: stream,
			text: () => Promise.resolve(""),
		})

		const reasoningHandler = new VertexHandler({
			...options,
			apiModelId: "gemini-2.5-flash-preview-05-20:thinking",
		})

		const messages = []
		for await (const msg of reasoningHandler.createMessage("sys", [])) {
			messages.push(msg)
		}

		// Should yield tool_call_partial, NOT reasoning
		const toolCallPartials = messages.filter((m) => m.type === "tool_call_partial")
		const reasoningParts = messages.filter((m) => m.type === "reasoning")

		expect(toolCallPartials.length).toBe(2) // name + arguments
		expect(toolCallPartials[0]).toMatchObject({ type: "tool_call_partial", name: "update_todo_list" })
		expect(reasoningParts.length).toBe(0)
	})

	it("should handle interleaved thinking with function calls", async () => {
		const chunks = [
			JSON.stringify({
				candidates: [
					{
						content: {
							role: "model",
							parts: [{ thought: true, text: "I need to create a todo list." }],
						},
					},
				],
			}),
			JSON.stringify({
				candidates: [
					{
						content: {
							role: "model",
							parts: [
								{
									functionCall: {
										name: "update_todo_list",
										args: { todos: "[ ] Task 1" },
									},
									thoughtSignature: "sig-456",
								},
							],
						},
					},
				],
			}),
			JSON.stringify({
				candidates: [
					{
						content: {
							role: "model",
							parts: [{ thought: true, text: "Good, todo list created." }],
						},
					},
				],
			}),
			JSON.stringify({
				candidates: [
					{
						content: {
							role: "model",
							parts: [{ text: "Done!" }],
						},
						finishReason: "STOP",
					},
				],
				usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 20 },
			}),
		]

		const encoder = new TextEncoder()
		const stream = new ReadableStream({
			start(controller) {
				chunks.forEach((chunk) => controller.enqueue(encoder.encode(chunk)))
				controller.close()
			},
		})

		fetchMock.mockResolvedValue({
			ok: true,
			body: stream,
			text: () => Promise.resolve(""),
		})

		const reasoningHandler = new VertexHandler({
			...options,
			apiModelId: "gemini-2.5-flash-preview-05-20:thinking",
		})

		const messages = []
		for await (const msg of reasoningHandler.createMessage("sys", [])) {
			messages.push(msg)
		}

		const reasoningParts = messages.filter((m) => m.type === "reasoning")
		const toolCallPartials = messages.filter((m) => m.type === "tool_call_partial")
		const textParts = messages.filter((m) => m.type === "text")

		// 2 reasoning blocks (from thought: true parts)
		expect(reasoningParts.length).toBe(2)
		expect(reasoningParts[0].text).toBe("I need to create a todo list.")
		expect(reasoningParts[1].text).toBe("Good, todo list created.")

		// 2 tool_call_partial (name + arguments for the function call)
		expect(toolCallPartials.length).toBe(2)
		expect(toolCallPartials[0]).toMatchObject({ name: "update_todo_list" })

		// 1 text block
		expect(textParts.length).toBe(1)
		expect(textParts[0].text).toBe("Done!")
	})

	it("should yield grounding sources at the end of the stream", async () => {
		const chunks = [
			JSON.stringify({
				candidates: [
					{
						content: { parts: [{ text: "Grounding info" }] },
						groundingMetadata: {
							groundingChunks: [{ web: { uri: "https://example.com", title: "Example" } }],
						},
					},
				],
			}),
		]

		const encoder = new TextEncoder()
		const stream = new ReadableStream({
			start(controller) {
				chunks.forEach((chunk) => controller.enqueue(encoder.encode(chunk)))
				controller.close()
			},
		})

		fetchMock.mockResolvedValue({
			ok: true,
			body: stream,
			text: () => Promise.resolve(""),
		})

		const messages = []
		for await (const msg of handler.createMessage("sys", [])) {
			messages.push(msg)
		}

		const groundingMsg = messages.find((m) => m.type === "grounding")
		expect(groundingMsg).toBeDefined()
		expect(groundingMsg?.sources).toEqual([{ title: "Example", url: "https://example.com" }])
	})
})
