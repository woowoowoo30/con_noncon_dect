window.onload = function() {
    fetch('/all/7_11')
      .then(response => response.json())
      .then(data => {
        // 處理 API 回傳的資料
        console.log(data);
      })
      .catch(error => console.error(error));
  }