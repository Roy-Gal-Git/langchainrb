# frozen_string_literal: true

module Langchain::LLM
  # LLM interface for Aws Bedrock APIs: https://docs.aws.amazon.com/bedrock/
  #
  # Gem requirements:
  #    gem 'aws-sdk-bedrockruntime', '~> 1.1'
  #
  # Usage:
  #    llm = Langchain::LLM::AwsBedrock.new(default_options: {})
  #
  class AwsBedrock < Base
    DEFAULTS = {
      chat_model: "anthropic.claude-3-5-sonnet-20240620-v1:0",
      completion_model: "anthropic.claude-v2:1",
      embedding_model: "amazon.titan-embed-text-v1",
      max_tokens_to_sample: 300,
      temperature: 1,
      top_k: 250,
      top_p: 0.999,
      stop_sequences: ["\n\nHuman:"],
      return_likelihoods: "NONE"
    }.freeze

    attr_reader :client, :defaults

    SUPPORTED_COMPLETION_PROVIDERS = %i[
      anthropic
      ai21
      cohere
      meta
    ].freeze

    SUPPORTED_CHAT_COMPLETION_PROVIDERS = %i[
      anthropic
      ai21
      mistral
    ].freeze

    SUPPORTED_EMBEDDING_PROVIDERS = %i[
      amazon
      cohere
    ].freeze

    def initialize(aws_client_options: {}, default_options: {})
      depends_on "aws-sdk-bedrockruntime", req: "aws-sdk-bedrockruntime"

      @client = ::Aws::BedrockRuntime::Client.new(**aws_client_options)
      @defaults = DEFAULTS.merge(default_options)

      chat_parameters.update(
        model: {default: @defaults[:chat_model]},
        temperature: {},
        max_tokens: {default: @defaults[:max_tokens_to_sample]},
        metadata: {},
        system: {}
      )
      chat_parameters.ignore(:n, :user)
      chat_parameters.remap(stop: :stop_sequences)
    end

    #
    # Generate an embedding for a given text
    #
    # @param text [String] The text to generate an embedding for
    # @param params extra parameters passed to Aws::BedrockRuntime::Client#invoke_model
    # @return [Langchain::LLM::AwsTitanResponse] Response object
    #
    def embed(text:, **params)
      raise "Completion provider #{embedding_provider} is not supported." unless SUPPORTED_EMBEDDING_PROVIDERS.include?(embedding_provider)

      parameters = compose_embedding_parameters params.merge(text:)

      response = client.invoke_model({
        model_id: @defaults[:embedding_model],
        body: parameters.to_json,
        content_type: "application/json",
        accept: "application/json"
      })

      parse_embedding_response response
    end

    #
    # Generate a completion for a given prompt
    #
    # @param prompt [String] The prompt to generate a completion for
    # @param params  extra parameters passed to Aws::BedrockRuntime::Client#invoke_model
    # @return [Langchain::LLM::Response::AnthropicResponse], [Langchain::LLM::Response::CohereResponse] or [Langchain::LLM::AI21Response] Response object
    #
    def complete(
      prompt:,
      model: @defaults[:completion_model],
      **params
    )
      raise "Completion provider #{model} is not supported." unless SUPPORTED_COMPLETION_PROVIDERS.include?(provider_name(model))

      parameters = compose_parameters(params, model)

      parameters[:prompt] = wrap_prompt prompt

      response = client.invoke_model({
        model_id: model,
        body: parameters.to_json,
        content_type: "application/json",
        accept: "application/json"
      })

      parse_response(response, model)
    end

    # Generate a chat completion for a given prompt
    # Currently only configured to work with the Anthropic provider and
    # the claude-3 model family
    #
    # @param [Hash] params unified chat parmeters from [Langchain::LLM::Parameters::Chat::SCHEMA]
    # @option params [Array<String>] :messages The messages to generate a completion for
    # @option params [String] :system The system prompt to provide instructions
    # @option params [String] :model The model to use for completion defaults to @defaults[:chat_model]
    # @option params [Integer] :max_tokens The maximum number of tokens to generate defaults to @defaults[:max_tokens_to_sample]
    # @option params [Array<String>] :stop The stop sequences to use for completion
    # @option params [Array<String>] :stop_sequences The stop sequences to use for completion
    # @option params [Float] :temperature The temperature to use for completion
    # @option params [Float] :top_p Use nucleus sampling.
    # @option params [Integer] :top_k Only sample from the top K options for each subsequent token
    # @yield [Hash] Provides chunks of the response as they are received
    # @return [Langchain::LLM::Response::AnthropicResponse] Response object
    def chat(params = {}, &block)
      parameters = chat_parameters.to_params(params)
      model_id = parameters[:model]

      unless SUPPORTED_CHAT_COMPLETION_PROVIDERS.include?(provider_name(model_id))
        raise "Chat provider #{model_id} is not supported."
      end

      converse_params = build_converse_params(parameters, model_id)

      if block
        response_chunks = []

        client.converse_stream(
          model_id: model_id,
          **converse_params
        ) do |stream|
          stream.on_event do |event|
            chunk = parse_converse_stream_event(event, model_id)
            response_chunks << chunk if chunk

            yield chunk if chunk
          end
        end

        response_from_converse_chunks(response_chunks)
      else
        response = client.converse(
          model_id: model_id,
          **converse_params
        )

        parse_converse_response(response, model_id)
      end
    end

    private

    def parse_model_id(model_id)
      model_id
        .gsub(/^(us|eu|apac|us-gov)\./, "") # Meta append "us." to their model ids, and AWS region prefixes are used by Bedrock cross-region model IDs.
        .split(".")
    end

    def provider_name(model_id)
      parse_model_id(model_id).first.to_sym
    end

    def model_name(model_id)
      parse_model_id(model_id).last
    end

    def completion_provider
      @defaults[:completion_model].split(".").first.to_sym
    end

    def embedding_provider
      @defaults[:embedding_model].split(".").first.to_sym
    end

    def wrap_prompt(prompt)
      if completion_provider == :anthropic
        "\n\nHuman: #{prompt}\n\nAssistant:"
      else
        prompt
      end
    end

    def max_tokens_key
      if completion_provider == :anthropic
        :max_tokens_to_sample
      elsif completion_provider == :cohere
        :max_tokens
      elsif completion_provider == :ai21
        :maxTokens
      end
    end

    def compose_parameters(params, model_id)
      if provider_name(model_id) == :anthropic
        compose_parameters_anthropic(params)
      elsif provider_name(model_id) == :cohere
        compose_parameters_cohere(params)
      elsif provider_name(model_id) == :ai21
        params
      elsif provider_name(model_id) == :meta
        params
      elsif provider_name(model_id) == :mistral
        params
      end
    end

    def compose_embedding_parameters(params)
      if embedding_provider == :amazon
        compose_embedding_parameters_amazon params
      elsif embedding_provider == :cohere
        compose_embedding_parameters_cohere params
      end
    end

    def parse_response(response, model_id)
      if provider_name(model_id) == :anthropic
        Langchain::LLM::Response::AnthropicResponse.new(JSON.parse(response.body.string))
      elsif provider_name(model_id) == :cohere
        Langchain::LLM::Response::CohereResponse.new(JSON.parse(response.body.string))
      elsif provider_name(model_id) == :ai21
        Langchain::LLM::Response::AI21Response.new(JSON.parse(response.body.string, symbolize_names: true))
      elsif provider_name(model_id) == :meta
        Langchain::LLM::Response::AwsBedrockMetaResponse.new(JSON.parse(response.body.string))
      elsif provider_name(model_id) == :mistral
        Langchain::LLM::Response::MistralAIResponse.new(JSON.parse(response.body.string))
      end
    end

    def parse_embedding_response(response)
      json_response = JSON.parse(response.body.string)

      if embedding_provider == :amazon
        Langchain::LLM::Response::AwsTitanResponse.new(json_response)
      elsif embedding_provider == :cohere
        Langchain::LLM::Response::CohereResponse.new(json_response)
      end
    end

    def compose_embedding_parameters_amazon(params)
      default_params = @defaults.merge(params)

      {
        inputText: default_params[:text],
        dimensions: default_params[:dimensions],
        normalize: default_params[:normalize]
      }.compact
    end

    def compose_embedding_parameters_cohere(params)
      default_params = @defaults.merge(params)

      {
        texts: [default_params[:text]],
        truncate: default_params[:truncate],
        input_type: default_params[:input_type],
        embedding_types: default_params[:embedding_types]
      }.compact
    end

    def compose_parameters_cohere(params)
      default_params = @defaults.merge(params)

      {
        max_tokens: default_params[:max_tokens_to_sample],
        temperature: default_params[:temperature],
        p: default_params[:top_p],
        k: default_params[:top_k],
        stop_sequences: default_params[:stop_sequences]
      }
    end

    def compose_parameters_anthropic(params)
      params.merge(anthropic_version: "bedrock-2023-05-31")
    end

    def build_converse_params(params, model_id)
      messages = params[:messages] || []
      system_prompt = params[:system]

      # Transform messages to Converse API format
      # Converse API expects messages with role and content array
      converse_messages = messages.map do |msg|
        content = normalize_message_content(msg[:content])
        {
          role: msg[:role],
          content: content
        }
      end

      # Build system content blocks if system prompt is provided
      system_blocks = system_prompt ? [{text: system_prompt}] : nil

      # Build inference config
      inference_config = {}
      inference_config[:max_tokens] = params[:max_tokens] || @defaults[:max_tokens_to_sample] if params[:max_tokens] || @defaults[:max_tokens_to_sample]
      inference_config[:temperature] = params[:temperature] if params[:temperature]
      inference_config[:top_p] = params[:top_p] if params[:top_p]
      inference_config[:top_k] = params[:top_k] if params[:top_k]
      stop_sequences = params[:stop_sequences] || params[:stop]
      inference_config[:stop_sequences] = stop_sequences if stop_sequences && !stop_sequences.empty?

      result = {
        messages: converse_messages
      }
      result[:system] = system_blocks if system_blocks
      result[:inference_config] = inference_config if inference_config.any?

      # Handle tools if provided
      if params[:tools] && !params[:tools].empty?
        result[:tool_config] = build_tool_config(params)
      end

      result
    end

    def normalize_message_content(content)
      # If content is a string, convert to content block format
      if content.is_a?(String)
        [{text: content}]
      # If content is already an array, ensure it's in the right format
      elsif content.is_a?(Array)
        content.map do |item|
          if item.is_a?(String)
            {text: item}
          elsif item.is_a?(Hash)
            # Handle different content block types
            if item[:type] == "text" || item["type"] == "text"
              {text: item[:text] || item["text"]}
            elsif item[:type] == "image" || item["type"] == "image"
              {
                image: {
                  format: item.dig(:source, :type) || item.dig("source", "type") || "base64",
                  source: {
                    bytes: item.dig(:source, :data) || item.dig("source", "data")
                  }
                }
              }
            elsif item[:type] == "tool_use" || item["type"] == "tool_use"
              {
                tool_use: {
                  tool_use_id: item[:id] || item["id"],
                  name: item[:name] || item["name"],
                  input: item[:input] || item["input"] || {}
                }
              }
            elsif item[:type] == "tool_result" || item["type"] == "tool_result"
              {
                tool_result: {
                  tool_use_id: item[:tool_use_id] || item["tool_use_id"],
                  content: normalize_message_content(item[:content] || item["content"] || [])
                }
              }
            else
              item
            end
          else
            item
          end
        end
      else
        [{text: content.to_s}]
      end
    end

    def build_tool_config(params)
      tools = params[:tools] || []
      tool_choice = params[:tool_choice]

      tool_config = {
        tools: tools.map do |tool|
          {
            tool_spec: {
              name: tool[:name] || tool["name"],
              description: tool[:description] || tool["description"],
              input_schema: tool[:parameters] || tool["parameters"] || {}
            }
          }
        end
      }

      # Handle tool_choice
      if tool_choice
        if tool_choice == "auto" || tool_choice == "any"
          tool_config[:tool_choice] = {auto: {}}
        elsif tool_choice == "none"
          tool_config[:tool_choice] = {none: {}}
        elsif tool_choice.is_a?(Hash) && (tool_choice[:type] == "function" || tool_choice["type"] == "function")
          tool_config[:tool_choice] = {
            tool: {
              name: tool_choice[:function]&.dig(:name) || tool_choice["function"]&.dig("name")
            }
          }
        end
      end

      tool_config
    end

    def parse_converse_response(response, model_id)
      # Transform Converse API response to AnthropicResponse format
      output = response.output
      raw_response = {
        "id" => response.id || "msg_#{Time.now.to_i}_#{rand(10000)}",
        "type" => "message",
        "role" => output.role || "assistant",
        "content" => output.content.map do |content_block|
          if content_block.text
            {type: "text", text: content_block.text}
          elsif content_block.tool_use
            {
              type: "tool_use",
              id: content_block.tool_use.tool_use_id,
              name: content_block.tool_use.name,
              input: content_block.tool_use.input || {}
            }
          else
            {type: "text", text: ""}
          end
        end,
        "model" => model_id,
        "stop_reason" => response.stop_reason,
        "usage" => {
          "input_tokens" => response.usage&.input_tokens || 0,
          "output_tokens" => response.usage&.output_tokens || 0
        }
      }

      Langchain::LLM::Response::AnthropicResponse.new(raw_response)
    end

    def parse_converse_stream_event(event, model_id = nil)
      # Parse Converse Stream API events
      case event
      when Aws::BedrockRuntime::Types::MessageStartEvent
        {
          "type" => "message_start",
          "message" => {
            "id" => event.message.id || "msg_#{Time.now.to_i}_#{rand(10000)}",
            "type" => "message",
            "role" => event.message.role || "assistant",
            "content" => [],
            "model" => model_id,
            "stop_reason" => nil,
            "usage" => {
              "input_tokens" => event.message.usage&.input_tokens || 0,
              "output_tokens" => event.message.usage&.output_tokens || 0
            }
          }
        }
      when Aws::BedrockRuntime::Types::ContentBlockStartEvent
        content_block = event.content_block
        content_block_hash = if content_block.text
          {type: "text", text: ""}
        elsif content_block.tool_use
          {
            type: "tool_use",
            id: content_block.tool_use.tool_use_id,
            name: content_block.tool_use.name,
            input: {}
          }
        else
          {type: "text", text: ""}
        end

        {
          "type" => "content_block_start",
          "index" => event.content_block_index,
          "content_block" => content_block_hash
        }
      when Aws::BedrockRuntime::Types::ContentBlockDeltaEvent
        delta = event.delta
        delta_hash = if delta.text
          {type: "text_delta", text: delta.text}
        elsif delta.tool_use
          {
            type: "input_json_delta",
            partial_json: delta.tool_use.partial_json || ""
          }
        else
          {}
        end

        {
          "type" => "content_block_delta",
          "index" => event.content_block_index,
          "delta" => delta_hash
        }
      when Aws::BedrockRuntime::Types::ContentBlockStopEvent
        {
          "type" => "content_block_stop",
          "index" => event.content_block_index
        }
      when Aws::BedrockRuntime::Types::MessageDeltaEvent
        delta = event.delta
        delta_hash = {
          "stop_reason" => delta.stop_reason,
          "stop_sequence" => delta.stop_sequence
        }

        result = {
          "type" => "message_delta",
          "delta" => delta_hash
        }
        result["usage"] = {
          "output_tokens" => event.usage&.output_tokens || 0
        } if event.usage

        result
      when Aws::BedrockRuntime::Types::MessageStopEvent
        metrics_hash = {}
        if event.metrics
          metrics_hash = {
            "inputTokenCount" => event.metrics.input_token_count || 0,
            "outputTokenCount" => event.metrics.output_token_count || 0,
            "invocationLatency" => event.metrics.latency&.invocation_latency || 0,
            "firstByteLatency" => event.metrics.latency&.first_byte_latency || 0
          }
        end

        {
          "type" => "message_stop",
          "amazon-bedrock-invocationMetrics" => metrics_hash
        }
      else
        nil
      end
    end

    def response_from_converse_chunks(chunks)
      # Build response from Converse Stream chunks (same format as invoke_model)
      response_from_chunks(chunks)
    end

    def response_from_chunks(chunks)
      raw_response = {}

      chunks.group_by { |chunk| chunk["type"] }.each do |type, chunks|
        case type
        when "message_start"
          raw_response = chunks.first["message"]
        when "content_block_start"
          raw_response["content"] = chunks.map { |chunk| chunk["content_block"] }
        when "content_block_delta"
          chunks.group_by { |chunk| chunk["index"] }.each do |index, deltas|
            deltas.group_by { |delta| delta.dig("delta", "type") }.each do |type, deltas|
              case type
              when "text_delta"
                raw_response["content"][index]["text"] = deltas.map { |delta| delta.dig("delta", "text") }.join
              when "input_json_delta"
                json_string = deltas.map { |delta| delta.dig("delta", "partial_json") }.join
                raw_response["content"][index]["input"] = json_string.empty? ? {} : JSON.parse(json_string)
              end
            end
          end
        when "message_delta"
          chunks.each do |chunk|
            raw_response = raw_response.merge(chunk["delta"])
            raw_response["usage"] = raw_response["usage"].merge(chunk["usage"]) if chunk["usage"]
          end
        end
      end

      Langchain::LLM::Response::AnthropicResponse.new(raw_response)
    end
  end
end
