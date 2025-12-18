# frozen_string_literal: true

module Langchain::LLM
  # Bedrock chat implementation via Converse / ConverseStream.
  #
  # This class intentionally targets the Bedrock Converse API (message-based),
  # and yields native AWS SDK ConverseStream events when a block is provided.
  #
  # For InvokeModel-based operations (complete/embed), use `Langchain::LLM::AwsBedrock`.
  class AwsBedrockConverse < Base
    MIN_BEDROCKRUNTIME_VERSION_FOR_CONVERSE = Gem::Version.new("1.68.0")

    DEFAULTS = {
      chat_model: "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
      max_tokens_to_sample: 300,
      temperature: 1,
      top_k: 250,
      top_p: 0.999,
      stop_sequences: ["\n\nHuman:"]
    }.freeze

    attr_reader :client, :defaults

    def initialize(aws_client_options: {}, default_options: {})
      depends_on "aws-sdk-bedrockruntime", req: "aws-sdk-bedrockruntime"

      bedrockruntime_spec = Gem.loaded_specs["aws-sdk-bedrockruntime"]
      if bedrockruntime_spec && bedrockruntime_spec.version < MIN_BEDROCKRUNTIME_VERSION_FOR_CONVERSE
        raise ArgumentError,
              "aws-sdk-bedrockruntime #{bedrockruntime_spec.version} is too old for Bedrock Converse API support. " \
              "Please upgrade to >= #{MIN_BEDROCKRUNTIME_VERSION_FOR_CONVERSE}."
      end

      @client = ::Aws::BedrockRuntime::Client.new(**aws_client_options)
      @defaults = DEFAULTS.merge(default_options)

      chat_parameters.update(
        model: {default: @defaults[:chat_model]},
        temperature: {},
        max_tokens: {default: @defaults[:max_tokens_to_sample]},
        system: {}
      )
      chat_parameters.ignore(:n, :user)
      chat_parameters.remap(stop: :stop_sequences)
    end

    # Generate a chat completion via Bedrock Converse API.
    #
    # When a block is provided, yields native AWS SDK ConverseStream events as they arrive.
    # Returns a `Langchain::LLM::AnthropicResponse` to keep Assistant behavior consistent.
    def chat(params = {}, &block)
      parameters = chat_parameters.to_params(params)
      model_id = parameters[:model]

      request = build_converse_request(parameters)

      if block
        builder = ConverseStreamResponseBuilder.new(model_id: model_id)

        # The converse_stream method blocks until all events are received and the stream is complete.
        # This ensures MessageStopEvent (with usage info) is received before we return.
        # The block will not return until the stream is fully consumed and closed.
        client.converse_stream(model_id: model_id, **request) do |stream|
          stream.on_event do |event|
            builder.consume(event)
            yield event
          end
        end

        # Only return the response after the stream is fully consumed
        builder.to_response
      else
        response = client.converse(model_id: model_id, **request)
        parse_converse_response(response, model_id)
      end
    end

    private

    def build_converse_request(parameters)
      messages = Array(parameters[:messages])
      system_param = parameters[:system]

      system_messages, non_system_messages = messages.partition { |msg| msg[:role] == "system" }

      system_blocks = system_messages.flat_map { |msg| normalize_message_content(msg[:content]) }
      system_blocks = system_blocks.concat([{text: system_param}]) if system_param && system_blocks.empty?

      converse_messages = non_system_messages.map do |msg|
        {
          role: msg[:role],
          content: normalize_message_content(msg[:content])
        }
      end

      inference_config = build_inference_config(parameters)

      result = {messages: converse_messages}
      result[:system] = system_blocks if system_blocks.any?
      result[:inference_config] = inference_config if inference_config.any?

      tool_config = build_tool_config(parameters)
      result[:tool_config] = tool_config if tool_config

      result
    end

    def build_inference_config(parameters)
      stop_sequences = parameters[:stop_sequences] || parameters[:stop]

      {}.tap do |cfg|
        cfg[:max_tokens] = parameters[:max_tokens] || @defaults[:max_tokens_to_sample]
        cfg[:temperature] = parameters[:temperature] if parameters.key?(:temperature)
        cfg[:top_p] = parameters[:top_p] if parameters.key?(:top_p)
        cfg[:top_k] = parameters[:top_k] if parameters.key?(:top_k)
        cfg[:stop_sequences] = stop_sequences if stop_sequences && !stop_sequences.empty?
      end
    end

    def build_tool_config(parameters)
      tools = Array(parameters[:tools])
      return nil if tools.empty?

      tool_specs =
        tools.map do |tool|
          # Expected: Anthropic tool schema {name:, description:, input_schema: {...}}
          name = tool[:name]
          description = tool[:description] || ""
          schema = tool[:input_schema] || {}

          {
            tool_spec: {
              name: name.to_s,
              description: description.to_s,
              input_schema: {json: schema}
            }
          }
        end

      cfg = {tools: tool_specs}

      cfg[:tool_choice] = tool_choice_to_bedrock(parameters[:tool_choice])

      cfg.compact
    end

    def tool_choice_to_bedrock(tool_choice)
      return nil if tool_choice.nil?

      if tool_choice.is_a?(Hash)
        type = tool_choice[:type] || tool_choice["type"]
        name = tool_choice[:name] || tool_choice["name"]

        return {auto: {}} if type == "auto"
        return {any: {}} if type == "any"
        return {none: {}} if type == "none"
        return {tool: {name: name.to_s}} if type == "tool"

        raise ArgumentError, "Unsupported tool_choice hash for Bedrock Converse: #{tool_choice.inspect}"
      end

      case tool_choice
      when "auto"
        {auto: {}}
      when "any"
        {any: {}}
      when "none"
        {none: {}}
      else
        {tool: {name: tool_choice.to_s}}
      end
    end

    def normalize_message_content(content)
      case content
      when String
        [{text: content}]
      when Array
        content.map { |item| normalize_content_block(item) }.compact
      when NilClass
        []
      else
        [{text: content.to_s}]
      end
    end

    def normalize_content_block(item)
      return {text: item} if item.is_a?(String)
      return item if !item.is_a?(Hash)

      type = item[:type] || item["type"]

      case type
      when "text"
        {text: item[:text] || item["text"]}
      when "tool_use"
        {
          tool_use: {
            tool_use_id: item[:id] || item["id"] || item[:tool_use_id] || item["tool_use_id"],
            name: item[:name] || item["name"],
            input: item[:input] || item["input"] || {}
          }
        }
      when "tool_result"
        {
          tool_result: {
            tool_use_id: item[:tool_use_id] || item["tool_use_id"],
            content: normalize_message_content(item[:content] || item["content"] || "")
          }
        }
      when "image"
        source = item[:source] || item["source"] || {}
        {
          image: {
            format: source[:type] || source["type"] || "base64",
            source: {bytes: source[:data] || source["data"]}
          }
        }
      else
        item
      end
    end

    def parse_converse_response(response, model_id)
      output = response.output

      raw_response = {
        "id" => response.id,
        "type" => "message",
        "role" => output.role,
        "content" => Array(output.content).map { |blk| bedrock_content_block_to_anthropic_hash(blk) },
        "model" => model_id,
        "stop_reason" => response.stop_reason,
        "usage" => {
          "input_tokens" => response.usage&.input_tokens.to_i,
          "output_tokens" => response.usage&.output_tokens.to_i
        }
      }

      Langchain::LLM::AnthropicResponse.new(raw_response)
    end

    def bedrock_content_block_to_anthropic_hash(block)
      if block.text
        {"type" => "text", "text" => block.text}
      elsif block.tool_use
        tu = block.tool_use
        {"type" => "tool_use", "id" => tu.tool_use_id, "name" => tu.name, "input" => tu.input || {}}
      else
        {"type" => "text", "text" => ""}
      end
    end

    class ConverseStreamResponseBuilder
      def initialize(model_id:)
        @model_id = model_id
        @message = {
          "id" => "msg_#{Time.now.to_i}_#{rand(10000)}",
          "type" => "message",
          "role" => "assistant",
          "content" => [],
          "model" => model_id,
          "stop_reason" => nil,
          "usage" => {"input_tokens" => 0, "output_tokens" => 0}
        }
        @tool_input_buffers = Hash.new { |h, k| h[k] = "" }
      end

      def consume(event)
        case event
        when Aws::BedrockRuntime::Types::MessageStartEvent
          @message["role"] = event.role if event.role
        when Aws::BedrockRuntime::Types::ContentBlockStartEvent
          idx = event.content_block_index
          start = event.start
          @message["content"][idx] = start_block_to_anthropic_hash(start)
        when Aws::BedrockRuntime::Types::ContentBlockDeltaEvent
          idx = event.content_block_index
          delta = event.delta
          apply_delta(idx, delta)
        when Aws::BedrockRuntime::Types::ConverseStreamMetadataEvent
          # Access usage from the event hash as per AWS SDK structure
          # Using to_h keeps this stable and avoids version-specific method checks
          usage = event.to_h[:usage]
          if usage
            # Use dig to safely access nested values (works with both hashes and structs)
            input_tokens = usage.dig(:input_tokens)
            output_tokens = usage.dig(:output_tokens)
            @message["usage"]["input_tokens"] = input_tokens.to_i if input_tokens
            @message["usage"]["output_tokens"] = output_tokens.to_i if output_tokens
          end
          @message["stop_reason"] ||= event.to_h[:stop_reason] if event.to_h.key?(:stop_reason)
        when Aws::BedrockRuntime::Types::MessageStopEvent
          @message["stop_reason"] ||= event.to_h[:stop_reason] if event.to_h.key?(:stop_reason)
          # Usage may be available in MessageStopEvent or may have been set in ConverseStreamMetadataEvent
          # Access usage from the event hash as per AWS SDK structure
          # Using to_h keeps this stable and avoids version-specific method checks
          usage = event.to_h[:usage]
          if usage
            # Use dig to safely access nested values (works with both hashes and structs)
            input_tokens = usage.dig(:input_tokens)
            output_tokens = usage.dig(:output_tokens)
            @message["usage"]["input_tokens"] = input_tokens.to_i if input_tokens
            @message["usage"]["output_tokens"] = output_tokens.to_i if output_tokens
          end
        end
      end

      def to_response
        if @message["stop_reason"].nil?
          has_tool_use = Array(@message["content"]).any? { |blk| blk.is_a?(Hash) && blk["type"] == "tool_use" }
          @message["stop_reason"] = has_tool_use ? "tool_use" : "end_turn"
        end

        Langchain::LLM::AnthropicResponse.new(@message)
      end

      private

      def start_block_to_anthropic_hash(start)
        h = start.to_h

        if h[:tool_use]
          tu = h[:tool_use]
          {
            "type" => "tool_use",
            "id" => tu[:tool_use_id],
            "name" => tu[:name],
            "input" => {}
          }
        else
          {"type" => "text", "text" => ""}
        end
      end

      def apply_delta(idx, delta)
        delta_h = delta.to_h

        if delta_h[:text]
          ensure_text_block(idx)
          @message["content"][idx]["text"] += delta_h[:text].to_s
          return
        end

        return unless delta_h[:tool_use]

        tool_delta = delta_h[:tool_use]
        fragment = tool_delta[:partial_json] || tool_delta[:input] || tool_delta[:input_json] || tool_delta[:json]

        return if fragment.nil?

        @tool_input_buffers[idx] += fragment.is_a?(String) ? fragment : JSON.generate(fragment)

        ensure_tool_use_block(idx)
        json_string = @tool_input_buffers[idx]
        @message["content"][idx]["input"] = json_string.empty? ? {} : JSON.parse(json_string)
      rescue JSON::ParserError
        # Keep buffering until we have valid JSON.
      end

      def ensure_text_block(idx)
        blk = @message["content"][idx]
        if blk.nil?
          @message["content"][idx] = {"type" => "text", "text" => ""}
        elsif blk["type"] != "text"
          @message["content"][idx] = {"type" => "text", "text" => ""}
        elsif blk["text"].nil?
          blk["text"] = ""
        end
      end

      def ensure_tool_use_block(idx)
        blk = @message["content"][idx]
        return if blk.is_a?(Hash) && blk["type"] == "tool_use"

        @message["content"][idx] = {"type" => "tool_use", "input" => {}}
      end
    end
  end
end


