from ethopy.cli import cli

def main():
    """
    Main entry point for the ethopy package.
    Delegates to the Click CLI implementation.
    """
    cli(prog_name="ethopy")

if __name__ == "__main__":
    main()