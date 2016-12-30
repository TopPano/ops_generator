function testThrownMsg(t, errMsg, fn, ...args) {
  t.throws(() => fn(...args), errMsg);
}

module.exports = {
  testThrownMsg
}
