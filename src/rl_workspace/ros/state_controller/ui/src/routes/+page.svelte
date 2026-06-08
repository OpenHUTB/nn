<script>
  // fetch data from the server endpoint
  let url = "http://localhost:8000";
  let stateData = null;

  function fetchStateData() {
    fetch(url + "/state")
      .then((response) => response.json())
      .then((data) => {
        stateData = data;
      });
  }

  //   function postRandomTrajectory() {
  //     fetch(url + "/controller/trajectory", {
  //       method: "POST",
  //       headers: {
  //         "Content-Type": "application/json"
  //       },
  //       body: JSON.stringify({
  //         // quaternion
  //         orientation: {},
  //         // point
  //         position: {
  //           x: Math.random(),
  //           y: Math.random(),
  //           z: Math.random()
  //         }
  //       })
  //     });
  //   }
  async function postRandomTrajectory() {
    // controller/trajectory_random
    await fetch(url + "/controller/trajectory_random");
  }
</script>

<h1>state controller</h1>
<p>work in progress!</p>
<p>See the <a href="{url}/redoc">API documentation</a> for more information.</p>

<button on:click={postRandomTrajectory}>click to post random trajectory</button>
<button on:click={fetchStateData}>click to update state</button>
{#if stateData}
  <pre>{JSON.stringify(stateData, null, 2)}</pre>
{/if}
