<main>
  <slot />
</main>

<style>
  main {
    margin-top: 1rem;
    padding-bottom: 3rem;
    margin: 0 auto;
    max-width: 992px;
    box-sizing: border-box;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen-Sans, Ubuntu,
      Cantarell, "Helvetica Neue", sans-serif;
  }

  @media (max-width: 992px) {
    main {
      padding: 0 0.5rem;
    }
  }
</style>
