from translator_by_speech.cli import parse_args, TranslationCLI


def main() -> None:
    """Main entry point for the application."""
    args = parse_args()
    cli = TranslationCLI()

    # Set languages if specified
    cli.set_languages(args.source, args.target)

    # Handle non-interactive commands
    if args.record is not None:
        file_path = cli.record_audio(duration=args.record)
        if args.process is None:  # Process the recording if no file is specified
            cli.process_audio_file(file_path)
    elif args.process:
        cli.process_audio_file(args.process)
    elif args.interactive or (not args.record and not args.process):
        # Interactive mode is default if no other actions specified
        cli.run()


if __name__ == "__main__":
    main()
